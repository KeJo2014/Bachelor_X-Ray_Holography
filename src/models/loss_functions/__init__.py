from .center_focused_tversky_loss import CenterFocusedTverskyLoss
from .radially_weighted_loss import RadiallyWeightedLoss
from .dice_loss import DiceLoss

LOSS_FUNCTIONS = [CenterFocusedTverskyLoss, RadiallyWeightedLoss, DiceLoss]
