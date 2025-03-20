from dataclasses import dataclass

@dataclass
class FVTemplate:  # Variable fragment template where CDRH3s are grafted to
    fv_h: str
    fv_l: str
    fv_h_cdr3: str   # CDRH3
    fv_h_cdr3_prefix: str
    fv_h_cdr3_suffix: str
    fv_l_cdr1: str   # CDRL1
    fv_l_cdr3: str   # CDRL3
    fv_h_cdr3_gen_prefix: str  # CDRH3 prefix, must include for generation
    fv_h_cdr3_gen_suffix: str  # CDRH3 suffix, must include for generation

