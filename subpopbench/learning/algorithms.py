import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from transformers import get_scheduler

from subpopbench.models import networks
from subpopbench.learning import joint_dro
from subpopbench.learning.optimizers import get_optimizers
from subpopbench.utils.misc import mixup_data


ALGORITHMS = [
    'ERM',
    # subgroup methods
    'GroupDRO',
    'IRM',
    'CVaRDRO',
    'JTT',
    'LfF',
    'LISA',
    'DFR',
    # data augmentation
    'Mixup',
    # domain generalization methods
    'MMD',
    'CORAL',
    # imbalanced learning methods
    'ReSample',
    'ReWeight',
    'SqrtReWeight',
    'CBLoss',
    'Focal',
    'LDAM',
    'BSoftmax',
    'CRT',
    'ReWeightCRT',
    'VanillaCRT'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a subgroup robustness algorithm.
    Subclasses should implement the following:
    - _init_model()
    - _compute_loss()
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_examples = num_examples

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step):
        """Perform one update step."""
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a

        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)

        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ERM, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.featurizer = networks.Featurizer(data_type, input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_model()

    def _init_model(self):
        self.clip_grad = (self.data_type == "text" and self.hparams["optimizer"] == "adamw")

        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.data_type == "text":
            self.network.zero_grad()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)


class GroupDRO(ERM):
    """
    Group DRO minimizes the error at the worst group [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(GroupDRO, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.register_buffer(
            "q", torch.ones(self.num_classes * self.num_attributes).cuda())

    def _compute_loss(self, i, x, y, a, step):
        losses = self.loss(self.predict(x), y)

        for idx_g, idx_samples in self.return_groups(y, a):
            self.q[idx_g] *= (self.hparams["groupdro_eta"] * losses[idx_samples].mean()).exp().item()

        self.q /= self.q.sum()

        loss_value = 0
        for idx_g, idx_samples in self.return_groups(y, a):
            loss_value += self.q[idx_g] * losses[idx_samples].mean()

        return loss_value


class ReSample(ERM):
    """Naive resample, with no changes to ERM, but enable balanced sampling in hparams"""


class ReWeight(ERM):
    """Naive inverse re-weighting"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ReWeight, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        assert len(grp_sizes) == num_classes * num_attributes
        grp_sizes = [x if x else np.inf for x in grp_sizes]
        per_grp_weights = 1 / np.array(grp_sizes)
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)

    def _compute_loss(self, i, x, y, a, step):
        losses = self.loss(self.predict(x), y)

        all_g = y * self.num_attributes + a
        loss_value = (self.weights_per_grp.type_as(losses)[all_g] * losses).mean()

        return loss_value


class SqrtReWeight(ReWeight):
    """Square-root inverse re-weighting"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(SqrtReWeight, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        assert len(grp_sizes) == num_classes * num_attributes
        grp_sizes = [x if x else np.inf for x in grp_sizes]
        per_grp_weights = 1 / np.sqrt(np.array(grp_sizes))
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)


class CBLoss(ReWeight):
    """Class-balanced loss, https://arxiv.org/pdf/1901.05555.pdf"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CBLoss, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        assert len(grp_sizes) == num_classes * num_attributes
        grp_sizes = [x if x else np.inf for x in grp_sizes]
        effective_num = 1. - np.power(self.hparams["beta"], grp_sizes)
        effective_num = np.array(effective_num)
        effective_num[effective_num == 1] = np.inf
        per_grp_weights = (1. - self.hparams["beta"]) / effective_num
        per_grp_weights = per_grp_weights / np.sum(per_grp_weights) * len(grp_sizes)
        self.weights_per_grp = torch.FloatTensor(per_grp_weights)


class Focal(ERM):
    """Focal loss, https://arxiv.org/abs/1708.02002"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Focal, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    @staticmethod
    def focal_loss(input_values, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def _compute_loss(self, i, x, y, a, step):
        return self.focal_loss(self.loss(self.predict(x), y), self.hparams["gamma"])


class LDAM(ERM):
    """LDAM loss, https://arxiv.org/abs/1906.07413"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(LDAM, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        assert len(grp_sizes) == num_classes * num_attributes
        # attribute-agnostic as modifying class-dependent margins
        class_sizes = [np.sum(grp_sizes[i * num_attributes:(i+1) * num_attributes]) for i in range(num_classes)]
        class_sizes = [x if x else np.inf for x in class_sizes]
        m_list = 1. / np.sqrt(np.sqrt(np.array(class_sizes)))
        m_list = m_list * (self.hparams["max_m"] / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)

    def _compute_loss(self, i, x, y, a, step):
        x = self.predict(x)
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, y.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :].type_as(x), index_float.transpose(0, 1).type_as(x))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        loss_value = F.cross_entropy(self.hparams["scale"] * output, y)

        return loss_value


class BSoftmax(ERM):
    """Balanced softmax, https://arxiv.org/abs/2007.10740"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(BSoftmax, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        assert len(grp_sizes) == num_classes * num_attributes
        # attribute-agnostic as modifying class-dependent margins
        class_sizes = [np.sum(grp_sizes[i * num_attributes:(i+1) * num_attributes]) for i in range(num_classes)]
        self.n_samples_per_cls = torch.FloatTensor(class_sizes)

    def _compute_loss(self, i, x, y, a, step):
        x = self.predict(x)
        spc = self.n_samples_per_cls.type_as(x)
        spc = spc.unsqueeze(0).expand(x.shape[0], -1)
        x = x + spc.log()
        loss_value = F.cross_entropy(input=x, target=y)

        return loss_value


class CRT(ERM):
    """Classifier re-training with balanced sampling during the second earning stage"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CRT, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")


class ReWeightCRT(ReWeight):
    """Classifier re-training with balanced re-weighting during the second earning stage"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ReWeightCRT, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")


class VanillaCRT(ERM):
    """Classifier re-training with normal (instance-balanced) sampling"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(VanillaCRT, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")


class DFR(ERM):
    """
    Classifier re-training with sub-sampled, group-balanced, held-out(validation) data and l1 regularization.
    Note that when attribute is unavailable in validation data, group-balanced reduces to class-balanced.
    https://openreview.net/pdf?id=Zb6c8A-Fghk
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(DFR, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                0.
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                0.
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean() + self.hparams['dfr_reg'] * torch.norm(self.classifier.weight, 1)


class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(IRM, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def _compute_loss(self, i, x, y, a, step):
        penalty_weight = self.hparams['irm_lambda'] \
            if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 1.0
        nll = 0.
        penalty = 0.

        logits = self.network(x)
        for idx_a, idx_samples in self.return_attributes(a):
            nll += F.cross_entropy(logits[idx_samples], y[idx_samples])
            penalty += self._irm_penalty(logits[idx_samples], y[idx_samples])
        nll /= len(a.unique())
        penalty /= len(a.unique())
        loss_value = nll + (penalty_weight * penalty)

        self.update_count += 1
        return loss_value


class Mixup(ERM):
    """Mixup of minibatch data"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Mixup, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    def _compute_loss(self, i, x, y, a, step):
        if self.data_type == "text":
            feats = self.featurizer(x)
            feats, yi, yj, lam = mixup_data(feats, y, self.hparams["mixup_alpha"], device="cuda")
            predictions = self.classifier(feats)
        else:
            x, yi, yj, lam = mixup_data(x, y, self.hparams["mixup_alpha"], device="cuda")
            predictions = self.predict(x)
        loss_value = lam * F.cross_entropy(predictions, yi) + (1 - lam) * F.cross_entropy(predictions, yj)
        return loss_value


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions using MMD
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams,
                 grp_sizes=None, gaussian=False):
        super(AbstractMMD, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    @staticmethod
    def my_cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def _compute_loss(self, i, x, y, a, step):
        all_feats = self.featurizer(x)
        outputs = self.classifier(all_feats)
        objective = F.cross_entropy(outputs, y)

        features = []
        for _, idx_samples in self.return_attributes(a):
            features.append(all_feats[idx_samples])

        penalty = 0.
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                penalty += self.mmd(features[i], features[j])

        if len(features) > 1:
            penalty /= (len(features) * (len(features) - 1) / 2)

        loss_value = objective + (self.hparams['mmd_gamma'] * penalty)
        return loss_value


class MMD(AbstractMMD):
    """MMD using Gaussian kernel"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(MMD, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, gaussian=True)


class CORAL(AbstractMMD):
    """MMD using mean and covariance difference"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CORAL, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes, gaussian=False)


class CVaRDRO(ERM):
    """
    DRO with CVaR uncertainty set
    https://arxiv.org/pdf/2010.05893.pdf
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CVaRDRO, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self._joint_dro_loss_computer = joint_dro.RobustLoss(hparams['joint_dro_alpha'], 0, "cvar")

    def _compute_loss(self, i, x, y, a, step):
        per_sample_losses = self.loss(self.predict(x), y)
        actual_loss = self._joint_dro_loss_computer(per_sample_losses)
        return actual_loss


class AbstractTwoStage(Algorithm):
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.stage1_model = ERM(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.first_stage_step_frac = hparams['first_stage_step_frac']
        self.switch_step = int(self.first_stage_step_frac * hparams['steps'])
        self.cur_model = self.stage1_model

        self.stage2_model = None    # implement in child classes

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch

        if step < self.switch_step:
            self.cur_model = self.stage1_model
            self.cur_model.train()
            loss = self.stage1_model._compute_loss(all_i, all_x, all_y, all_a, step)
        else:
            self.cur_model = self.stage2_model
            self.cur_model.train()
            self.stage1_model.eval()
            loss = self.stage2_model._compute_loss(all_i, all_x, all_y, all_a, step, self.stage1_model)
        
        self.cur_model.optimizer.zero_grad()
        loss.backward()
        if self.cur_model.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.cur_model.network.parameters(), 1.0)
        self.cur_model.optimizer.step()

        if self.cur_model.lr_scheduler is not None:
            self.cur_model.lr_scheduler.step()

        if self.data_type == "text":
            self.cur_model.network.zero_grad()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.cur_model.featurizer(x)
    
    def predict(self, x):
        return self.cur_model.network(x)


class JTT_Stage2(ERM): 
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    def _compute_loss(self, i, x, y, a, step, stage1_model):
        with torch.no_grad():
            predictions = stage1_model.predict(x)

        if predictions.squeeze().ndim == 1:
            wrong_predictions = (predictions > 0).detach().ne(y).float()
        else:
            wrong_predictions = predictions.argmax(1).detach().ne(y).float()

        weights = torch.ones(wrong_predictions.shape).to(x.device).float()
        weights[wrong_predictions == 1] = self.hparams["jtt_lambda"]

        return (self.loss(self.predict(x), y) * weights).mean()


class JTT(AbstractTwoStage):
    """
    Just-train-twice (JTT) [https://arxiv.org/pdf/2107.09044.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.stage2_model = JTT_Stage2(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)


class LfF(Algorithm):
    """
    Learning from Failure (LfF) [https://arxiv.org/pdf/2007.02561.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.pred_model = ERM(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None)        

        self.biased_featurizer = networks.Featurizer(data_type, input_shape, self.hparams)
        self.biased_classifier = networks.Classifier(
            self.biased_featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.biased_network = nn.Sequential(self.biased_featurizer, self.biased_classifier)
        self.q = self.hparams['LfF_q']
        self._init_model()

    def _init_model(self):
        self.pred_model._init_model()

        self.clip_grad = (self.data_type == "text" and self.hparams["optimizer"] == "adamw")

        if self.data_type in ["images", "tabular"]:
            self.optimizer_b = get_optimizers['sgd'](
                self.biased_network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.biased_network.zero_grad()
            self.optimizer_b = get_optimizers[self.hparams["optimizer"]](
                self.biased_network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer_b,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    # implemented from equation
    def GCE(self, logits, targets):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = (1 - Yg.squeeze()**self.q) / self.q
        return loss

    # copied from the authors' repo
    def GCE2(self, logits, targets):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = F.cross_entropy(logits, targets, reduction='none') * (Yg.squeeze().detach()**self.q)*self.q
        return loss

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch    
        pred_logits = self.pred_model.predict(all_x) 
        biased_logits = self.biased_network(all_x)
        loss_gce = self.GCE2(biased_logits, all_y)
        ce_b = F.cross_entropy(biased_logits, all_y, reduction='none')
        ce_d = F.cross_entropy(pred_logits, all_y, reduction='none')
        weights = (ce_b/(ce_b + ce_d + 1e-8)).detach()

        self.optimizer_b.zero_grad()
        self.pred_model.optimizer.zero_grad()

        loss_pred = (ce_d * weights).mean()
        loss = loss_pred.mean() + loss_gce.mean()
        loss.backward()

        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.biased_network.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.pred_model.parameters(), 1.0)
        self.optimizer_b.step()
        self.pred_model.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.pred_model.lr_scheduler.step()

        if self.data_type == "text":
            self.biased_network.zero_grad()
            self.pred_model.zero_grad()

        return {'loss': loss.item(), 'loss_pred': loss_pred.mean().item(), 'loss_gce': loss_gce.mean().item()}

    def return_feats(self, x):
        return self.pred_model.featurizer(x)

    def predict(self, x):
        return self.pred_model.predict(x)


class LISA(ERM):
    """
    Improving Out-of-Distribution Robustness via Selective Augmentation [https://arxiv.org/pdf/2201.00299.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    def _to_ohe(self, y):
        return F.one_hot(y, num_classes=self.num_classes)

    def _lisa_mixup_data(self, s, a, x, y, alpha):
        if (not self.data_type == "images") or self.hparams['LISA_mixup_method'] == 'mixup':
            fn = self._mix_up
        elif self.hparams['LISA_mixup_method'] == 'cutmix':
            fn = self._cut_mix_up

        all_mix_x, all_mix_y = [], []
        bs = len(x)
        # repeat until enough samples
        while sum(list(map(len, all_mix_x))) < bs:
            start_len = sum(list(map(len, all_mix_x)))
            # same label, mixup between attributes
            if s:
                # can't do intra-label mixup with only one attribute
                if len(torch.unique(a)) < 2:
                    return x, y

                for y_i in range(self.num_classes):
                    mask = y[:, y_i].squeeze().bool()
                    x_i, y_i, a_i = x[mask], y[mask], a[mask]
                    unique_a_is = torch.unique(a_i)
                    if len(unique_a_is) < 2:
                        continue

                    # if there are multiple attributes, choose a random pair
                    a_i1, a_i2 = unique_a_is[torch.randperm(len(unique_a_is))][:2]
                    mask2_1 = a_i == a_i1
                    mask2_2 = a_i == a_i2
                    all_mix_x_i, all_mix_y_i = fn(alpha, x_i[mask2_1], x_i[mask2_2], y_i[mask2_1], y_i[mask2_2])
                    all_mix_x.append(all_mix_x_i)
                    all_mix_y.append(all_mix_y_i)

            # same attribute, mixup between labels
            else:
                # can't do intra-attribute mixup with only one label
                if len(y.sum(axis=0).nonzero()) < 2:
                    return x, y

                for a_i in torch.unique(a):
                    mask = a == a_i
                    x_i, y_i = x[mask], y[mask]
                    unique_y_is = y_i.sum(axis=0).nonzero()
                    if len(unique_y_is) < 2:
                        continue

                    # if there are multiple labels, choose a random pair
                    y_i1, y_i2 = unique_y_is[torch.randperm(len(unique_y_is))][:2] 
                    mask2_1 = y_i[:, y_i1].squeeze().bool()
                    mask2_2 = y_i[:, y_i2].squeeze().bool()
                    all_mix_x_i, all_mix_y_i = fn(alpha, x_i[mask2_1], x_i[mask2_2], y_i[mask2_1], y_i[mask2_2])
                    all_mix_x.append(all_mix_x_i)
                    all_mix_y.append(all_mix_y_i)

            end_len = sum(list(map(len, all_mix_x)))
            # each attribute only has one unique label
            if end_len == start_len:
                return x, y

        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)

        shuffle_idx = torch.randperm(len(all_mix_x))
        return all_mix_x[shuffle_idx][:bs], all_mix_y[shuffle_idx][:bs]

    @staticmethod
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    @staticmethod
    def _mix_up(alpha, x1, x2, y1, y2):
        # y1, y2 should be one-hot label, which means the shape of y1 and y2 should be [bsz, n_classes]
        length = min(len(x1), len(x2))
        x1 = x1[:length]
        x2 = x2[:length]
        y1 = y1[:length]
        y2 = y2[:length]

        n_classes = y1.shape[1]
        bsz = len(x1)
        l = np.random.beta(alpha, alpha, [bsz, 1])
        if len(x1.shape) == 4:
            l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
        else:
            l_x = np.tile(l, (1, *x1.shape[1:]))
        l_y = np.tile(l, [1, n_classes])

        # mixed_input = l * x + (1 - l) * x2
        mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
        mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2

        return mixed_x, mixed_y

    def _cut_mix_up(self, alpha, x1, x2, y1, y2):
        length = min(len(x1), len(x2))
        x1 = x1[:length]
        x2 = x2[:length]
        y1 = y1[:length]
        y2 = y2[:length]

        input = torch.cat([x1, x2])
        target = torch.cat([y1, y2])

        rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])

        lam = np.random.beta(alpha, alpha)
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

        return input, lam * target_a + (1-lam) * target_b

    def _compute_loss(self, i, x, y, a, step):
        s = np.random.random() <= self.hparams['LISA_p_sel']
        y_ohe = self._to_ohe(y)
        if self.data_type == "text":
            feats = self.featurizer(x)
            mixed_feats, mixed_y = self._lisa_mixup_data(s, a, feats, y_ohe, self.hparams["LISA_alpha"])
            predictions = self.classifier(mixed_feats)
        else:
            mixed_x, mixed_y = self._lisa_mixup_data(s, a, x, y_ohe, self.hparams["LISA_alpha"])
            predictions = self.predict(mixed_x)

        mixed_y_float = mixed_y.type(torch.FloatTensor)
        loss_value = F.cross_entropy(predictions, mixed_y_float.to(predictions.device))
        return loss_value
