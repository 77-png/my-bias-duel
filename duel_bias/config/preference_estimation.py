"""Compare two estimation of arm preference
"""

class WeightBiasInv:
    type = 'weight-bias_inv'
    name = 'Weighted-Voting'

class WeightIdentity:
    type = 'weight-identity'
    name = 'Id'

class WeightIdentityN:
    type = 'weight-identity-N'
    name = 'Minimum-Deviation'

class Config:
    type = "barp"  # Use BARP model

Config = WeightIdentityN