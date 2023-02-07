from ProposalSelection import ProposalSelection
from deepmask_pytorch.computeProposals import p_generation_test
from PairwiseSuperGlue import PairwiseSuperglue
from Integrated_scoremap_generation import Integrated_scoremap_generation
from ConvCRF.crf import CRF


def ProposalSuperglue(I,S_p,S): # I = image

    P = p_generation_test(img = I)
    P_s = ProposalSelection(P,S)
    M = PairwiseSuperglue(I, S_p, P, P_s)
    S_in = Integrated_scoremap_generation(I, S_p, P, M)
    pred,_ = CRF(I, S_in)

    return pred

