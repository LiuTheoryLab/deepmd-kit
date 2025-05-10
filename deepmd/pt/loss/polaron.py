# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Union,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from functools import partial

log = logging.getLogger(__name__)

class PolaronLoss(TaskLoss):
    def __init__(
        self,
        loss_func: str = "smooth_mae",
        metric: list = ["mae"],
        starter_learning_rate: float=1.0,
        start_pref_m: float = 1.00,
        limit_pref_m: float = 1.00,
        start_pref_t: float = 1.00,
        limit_pref_t: float = 1.00,
        beta: float = 1.00,
        **kwargs,
    ) -> None:
        r"""Construct a layer to compute loss on property.

        Parameters
        ----------
        task_dim : float
            The output dimension of property fitting net.
        var_name : str
            The atomic property to fit, 'energy', 'dipole', and 'polar'.
        loss_func : str
            The loss function, such as "smooth_mae", "mae", "rmse".
        metric : list
            The metric such as mae, rmse which will be printed.
        starter_learning_rate : float
            The learning rate for the model.
        start_pref_m : float
            The starting value for pref_m.
        limit_pref_m : float
            The limit value for pref_m.
        start_pref_t : float
            The starting value for pref_t.
        limit_pref_t : float
            The limit value for pref_t.
        beta : float
            The 'beta' parameter in 'smooth_mae' loss.
        """
        super().__init__()
        self.task_dim = 2            # alpha and beta channels
        self.var_name = "atom_spin"
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta

        self.starter_learning_rate = starter_learning_rate
        self.start_pref_m = start_pref_m
        self.limit_pref_m = limit_pref_m
        self.start_pref_t = start_pref_t
        self.limit_pref_t = limit_pref_t

        assert (
            self.start_pref_m >= 0.0
            and self.start_pref_m >= 0.0
            and self.limit_pref_m >= 0.0
            and self.limit_pref_m >= 0.0
        ), "Can't assign negative value to `pref_m` or `pref_t`"

    def forward(self, input_dict, model, label, natoms, learning_rate=0.0, mae=False):
        """Return loss on properties .

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        model_pred: dict[str, torch.Tensor]
            Model predictions.
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """
        model_pred = model(**input_dict)
        
        coef = learning_rate / self.starter_learning_rate
        pref_m = self.start_pref_m + (self.limit_pref_m - self.start_pref_m) * coef
        pref_t = self.start_pref_t + (self.limit_pref_t - self.start_pref_t) * coef

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        # get the label and model prediction
        spin_pred = model_pred["spin"]
        spin_label = label["atom_spin"].reshape([-1, natoms, self.task_dim])
        m_pred = spin_pred[:, 0] - spin_pred[:, 1]
        m_label = spin_label[:, 0] - spin_label[:, 1]
        M_pred = torch.sum(m_pred)
        M_label = torch.sum(m_label)

        # define the loss function
        if self.loss_func == "smooth_mae":
            loss_func = partial(F.smooth_l1_loss, reduction="sum", beta=self.beta)
        elif self.loss_func == "mae":
            loss_func = partial(F.l1_loss, reduction="sum")
        elif self.loss_func == "mse" :
            loss_func = partial(F.mse_loss, reduction="sum")
        elif self.loss_func == "rmse":
            loss_func = partial(F.mse_loss, reduction="mean")
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")
        
        # calculate the loss
        m_loss = loss_func(
            input=m_pred,
            target=m_label
        )
        spin_loss = loss_func(
            input=spin_pred,
            target=spin_label
        )
        M_loss = loss_func(
            input=M_pred,
            target=M_label
        )
        if self.loss_func == "rmse":
            m_loss = torch.sqrt(m_loss)
            spin_loss = torch.sqrt(spin_loss)
            M_loss = torch.sqrt(M_loss)
        loss += pref_t * ( m_loss + spin_loss) + pref_m * M_loss

        # more loss
        if "smooth_mae" in self.metric:
            loss_func = partial(F.smooth_l1_loss, reduction="mean", beta=self.beta)
            m_loss = loss_func(
                input=m_pred,
                target=m_label
            ).detach()
            spin_loss = loss_func(
                input=spin_pred,
                target=spin_label
            ).detach()
            M_loss = loss_func(
                input=M_pred,
                target=M_label
            ).detach()
            more_loss["m_smooth_mae"] = m_loss
            more_loss["spin_smooth_mae"] = spin_loss
            more_loss["M_smooth_mae"] = M_loss
        if "mae" in self.metric:
            loss_func = partial(F.l1_loss, reduction="mean")
            m_loss = loss_func(
                input=m_pred,
                target=m_label
            ).detach()
            spin_loss = loss_func(
                input=spin_pred,
                target=spin_label
            ).detach()
            M_loss = loss_func(
                input=M_pred,
                target=M_label
            ).detach()
            more_loss["m_mae"] = m_loss
            more_loss["spin_mae"] = spin_loss
            more_loss["M_mae"] = M_loss
        if "mse" in self.metric:
            loss_func = partial(F.mse_loss, reduction="mean")
            m_loss = loss_func(
                input=m_pred,
                target=m_label
            ).detach()
            spin_loss = loss_func(
                input=spin_pred,
                target=spin_label
            ).detach()
            M_loss = loss_func(
                input=M_pred,
                target=M_label
            ).detach()
            more_loss["m_mse"] = m_loss
            more_loss["spin_mse"] = spin_loss
            more_loss["M_mse"] = M_loss
        if "rmse" in self.metric:
            loss_func = partial(F.mse_loss, reduction="mean")
            m_loss = torch.sqrt(loss_func(
                input=m_pred,
                target=m_label
            )).detach()
            spin_loss = torch.sqrt(loss_func(
                input=spin_pred,
                target=spin_label
            )).detach()
            M_loss = torch.sqrt(loss_func(
                input=M_pred,
                target=M_label
            )).detach()
            more_loss["m_rmse"] = m_loss
            more_loss["spin_rmse"] = spin_loss
            more_loss["M_rmse"] = M_loss

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                'atomic_spin',
                ndof=self.task_dim,
                atomic=True,
                must=True,
                high_prec=True,
            )
        )
        return label_requirement
