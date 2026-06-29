from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.MISA import MISA
from .FusionNets.MULT import MULT
from .FusionNets.DUMIR import DUMIR
from .FusionNets.NMFIR import NMFIR

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder,
                    'bert-large-uncased': BERTEncoder
                }

methods_map = {
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mult': MULT,
    'dumir': DUMIR,
    'nmfir': NMFIR,
}