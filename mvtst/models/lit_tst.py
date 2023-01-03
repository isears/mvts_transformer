"""
Pytorch lightning implementation
"""
import pytorch_lightning as pl

from mvtst.models.ts_transformer import TSTransformerEncoderClassiregressor


class LitTST(pl.LightningModule):
    def __init__(self, tst: TSTransformerEncoderClassiregressor) -> None:
        super().__init__()
        # TODO
