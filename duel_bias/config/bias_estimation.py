"""Compare different estimation of user bias
"""

class OptimPref:
    type = "optim-from-pref"
    name = "OPT-P"

class CondPref:
    type = "cond-from-pref"

class WCondPref:
    type = "wcond-from-pref"
    name = "COND-P"

class OptimMeanPref:
    type = "optim-from-mean-pref"
    name = "OPT-MP"

class CondMeanPref:
    type = "cond-from-mean-pref"

class WCondMeanPref:
    type = "wcond-from-mean-pref"
    name = "COND-MP"

class OptimBiasMeanPref:
    type = "optim-from-bias-mean-pref"

class CondBiasMeanPref:
    type = "cond-from-bias-mean-pref"

class WCondBiasMeanPref:
    type = "wcond-from-bias-mean-pref"

class OptimMix:
    type = "optim-mix"
    name = "OPT-BMP(*)"

class CondMix:
    type = "cond-mix"

class WCondMix:
    type = "wcond-mix"
    name = "COND-BMP"
    
class Config:
    type = "barp"  # Use BARP model


Config = OptimMix