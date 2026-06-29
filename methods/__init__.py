from .TEXT.manager import TEXT
from .MISA.manager import MISA
from .MULT.manager import MULT
from .MAG_BERT.manager import MAG_BERT
from .DUMIR.manager import DUMIR
from .NMFIR.manager import NMFIR

method_map = {
    'text': TEXT,
    'misa': MISA,
    'mult': MULT,
    'mag_bert': MAG_BERT,
    'dumir': DUMIR,
    'nmfir': NMFIR
}
