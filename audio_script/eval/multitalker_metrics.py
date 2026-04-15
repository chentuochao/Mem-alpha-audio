import itertools
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import editdistance
import jiwer
import numpy as np
import pandas as pd
import torch
import string

def normalize_string(input_string):
    result = input_string.lower()
    # return result
    for char in string.punctuation:
        result = result.replace(char, "")
    return result




def compute_der(pred, gt, frame_duration=0.04, collar_frames=0):
    """
    Compute Diarization Error Rate (DER) between ground truth and prediction matrices.
    Args:
        gt: np.ndarray, shape (N, T), ground truth speaker activity (binary)
        pred: np.ndarray, shape (N, T), predicted speaker activity (binary)
        frame_duration: float, duration of each frame in seconds
        collar_frames: int, number of frames to ignore at segment boundaries
    Returns:
        der: float, overall DER
        details: dict, with 'miss', 'fa', 'conf', 'total' in seconds
    """
    N, T = gt.shape
    # Hungarian mapping: maximize overlap between gt and pred speakers
    cost = torch.tensor(-np.dot(gt, pred.T).astype(np.float32))
    row_ind, col_ind = linear_sum_assignment(cost)
    row_ind, col_ind = row_ind.cpu().numpy(), col_ind.cpu().numpy()
    pred_aligned = pred[col_ind]
    # Optionally apply collar (not shown here for brevity)
    # Compute errors
    ref_speech = gt.sum(axis=0) > 0
    sys_speech = pred_aligned.sum(axis=0) > 0
    # miss: Missed Detection, speaker in the ground truth but not predicted as active
    miss = np.logical_and(ref_speech, ~sys_speech).sum()
    # fa: False Alarm, no speaker in the ground truth but predicted as active
    fa = np.logical_and(~ref_speech, sys_speech).sum()
    # Confusion: frames where both are active, but speaker assignment is wrong
    speech_both = np.logical_and(ref_speech, sys_speech)
    speaker_err = np.sum(gt != pred_aligned, axis = 0)
    conf = np.sum(speech_both & (speaker_err > 0))

    acc_err = np.sum(gt != pred_aligned)/(gt.shape[0]*gt.shape[1])

    total_ref = ref_speech.sum()
    miss_sec = miss * frame_duration
    fa_sec = fa * frame_duration
    conf_sec = conf * frame_duration
    total_sec = total_ref * frame_duration
    der = (miss_sec + fa_sec + conf_sec) / total_sec if total_sec > 0 else 0.0
    return der, {"col_ind": col_ind, 'miss': miss_sec, 'fa': fa_sec, 'conf': conf_sec, 'total': total_sec, "acc_err": acc_err}


def _der_for_permutation(pred_aligned, gt, frame_duration):
    """Shared DER arithmetic given an already-aligned prediction matrix."""
    ref_speech = gt.sum(axis=0) > 0
    sys_speech = pred_aligned.sum(axis=0) > 0
    miss = np.logical_and(ref_speech, ~sys_speech).sum()
    fa = np.logical_and(~ref_speech, sys_speech).sum()
    speech_both = np.logical_and(ref_speech, sys_speech)
    speaker_err = np.sum(gt != pred_aligned, axis=0)
    conf = np.sum(speech_both & (speaker_err > 0))
    acc_err = np.sum(gt != pred_aligned) / (gt.shape[0] * gt.shape[1])
    total_ref = ref_speech.sum()
    miss_sec = miss * frame_duration
    fa_sec = fa * frame_duration
    conf_sec = conf * frame_duration
    total_sec = total_ref * frame_duration
    der = (miss_sec + fa_sec + conf_sec) / total_sec if total_sec > 0 else 0.0
    return der, {'miss': miss_sec, 'fa': fa_sec, 'conf': conf_sec, 'total': total_sec, 'acc_err': acc_err}


def compute_der_bruteforce(pred, gt, frame_duration=0.04, collar_frames=0):
    """
    Compute DER by exhaustively searching all permutations of pred rows.

    Picks the permutation that minimises DER.  Complexity is O(N! * T) so
    this should only be used when the number of speakers N is small.

    Args / Returns: same as ``compute_der``. also return the best permutation index
    """
    N_pred = pred.shape[0]
    N_gt = gt.shape[0]
    best_der = float('inf')
    best_details = None

    for perm in permutations(range(N_pred), N_gt):
        pred_aligned = pred[list(perm)]
        der, details = _der_for_permutation(pred_aligned, gt, frame_duration)
        if der < best_der:
            best_der = der
            best_details = details
            best_details["col_ind"] = np.array(perm)

    return best_der, best_details


def permutation_invariant_accuracy(x, y):
    # x and y: shape (M, T)
    M, T = x.shape
    # Compute pairwise accuracy matrix
    accuracy_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            accuracy_matrix[i, j] = np.mean(x[i] == y[j])
    # Hungarian algorithm (maximize accuracy)
    row_ind, col_ind = linear_sum_assignment(-accuracy_matrix)  # negative for maximization
    # Compute mean accuracy over optimal matching
    matched_accuracies = accuracy_matrix[row_ind, col_ind]
    return np.mean(matched_accuracies), row_ind, col_ind

def plot_diarout(gts, preds, figname):
    preds_mat = preds.cpu().numpy()#.transpose()
    gts_mat = gts.cpu().numpy()
    cmap_str, grid_color_p= 'viridis', 'gray'
    LW, FS = 0.2, 10

    yticklabels = ["spk0", "spk1", "spk2", "spk3"]
    yticks = np.arange(len(yticklabels))
    fig, axs = plt.subplots(2, 1, figsize=(100, 2))

    axs[0].imshow(gts_mat, cmap=cmap_str, interpolation='nearest')
    axs[0].set_title('GTs', fontsize=FS)
    axs[0].set_xticks(np.arange(-.5, gts_mat.shape[1], 1), minor=True)
    axs[0].set_yticks(yticks)
    axs[1].set_yticklabels(yticklabels)
    # axs[1].set_xticklabels(np.arange(-.5, preds_mat.shape[1], 1)/25.0)
    # axs[0].set_xlabel(f"40 ms Frames", fontsize=FS)
    axs[0].grid(which='minor', color=grid_color_p, linestyle='-', linewidth=LW)

    axs[1].imshow(preds_mat, cmap=cmap_str, interpolation='nearest')
    axs[1].set_title('Predictions', fontsize=FS)
    axs[1].set_xticks(np.arange(-.5, preds_mat.shape[1], 1), minor=True)
    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels(yticklabels)
    # axs[1].set_xticklabels(np.arange(-.5, preds_mat.shape[1], 1)/25.0)
    axs[1].set_xlabel(f"40 ms Frames", fontsize=FS)
    axs[1].grid(which='minor', color=grid_color_p, linestyle='-', linewidth=LW)

    plt.savefig(figname, dpi=300)
    plt.show()


@torch.jit.script
def unravel_index(index: int, shape: torch.Tensor):
    """
    Unravel the index input to fit the given shape.
    This function is needed for torch.jit.script compatibility.

    Args:
        index (int): The index to unravel.
        shape (Tesnor): The shape to unravel the index to.

    Returns:
        Tensor: The unraveled index.
    """
    out = []
    shape = torch.flip(shape, dims=(0,))
    for dim in shape:
        out.append(index % dim)
        index = index // dim
    out = torch.tensor([int(x.item()) for x in out])
    return torch.flip(out, dims=(0,))


@torch.jit.script
class LinearSumAssignmentSolver(object):
    """
    A Solver class for the linear sum assignment (LSA) problem.
    Designed for torch.jit.script compatibility in NeMo.

    The LSA problem is also referred to as bipartite matching problem. An LSA problem is described
    by a matrix `cost_mat`, where each cost_mat[i,j] is the cost of matching vertex i of the first partite
    set (e.g. a "worker") and vertex j of the second set (e.g. a "job").

    Thus, the goal of LSA-solver is to find a complete assignment of column element to row element with
    the minimal cost. Note that the solution may not be unique and there could be multiple solutions that
    yield the same minimal cost.

    LSA problem solver is needed for the following tasks in NeMo:
        - Permutation Invariant Loss (PIL) for diarization model training
        - Label permutation matching for online speaker diarzation
        - Concatenated minimum-permutation Word Error Rate (cp-WER) calculation

    This implementation is based on the LAP solver from scipy:
        https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py
        The scipy implementation comes with the following license:

        Copyright (c) 2008 Brian M. Clapper <bmc@clapper.org>, Gael Varoquaux
        Author: Brian M. Clapper, Gael Varoquaux
        License: 3-clause BSD

    References
        1. http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
        2. https://en.wikipedia.org/wiki/Hungarian_algorithm
        3. https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py


    Attributes:
        cost_mat (Tensor): 2D matrix containing cost matrix. Number of columns must be larger than number of rows.
        row_uncovered (Tensor): 1D matrix containing boolean values indicating whether a row is covered.
        col_uncovered (Tensor): 1D matrix containing boolean values indicating whether a column is covered.
        zero_row (Tensor): 1D matrix containing the row index of the last zero found.
        zero_col (Tensor): 1D matrix containing the column index of the last zero found.
        path (Tensor): 2D matrix containing the path taken through the matrix.
        marked (Tensor): 2D matrix containing the marked zeros.
    """

    def __init__(self, cost_matrix: torch.Tensor):
        # The main cost matrix
        self.cost_mat = cost_matrix
        row_len, col_len = self.cost_mat.shape

        # Initialize the solver state
        self.zero_row = torch.tensor(0, dtype=torch.long).to(cost_matrix.device)
        self.zero_col = torch.tensor(0, dtype=torch.long).to(cost_matrix.device)

        # Initialize the covered matrices
        self.row_uncovered = torch.ones(row_len, dtype=torch.bool).to(
            cost_matrix.device
        )
        self.col_uncovered = torch.ones(col_len, dtype=torch.bool).to(
            cost_matrix.device
        )

        # Initialize the path matrix and the mark matrix
        self.path = torch.zeros((row_len + col_len, 2), dtype=torch.long).to(
            cost_matrix.device
        )
        self.marked = torch.zeros((row_len, col_len), dtype=torch.long).to(
            cost_matrix.device
        )

    def _reset_uncovered_mat(self):
        """
        Clear all covered matrix cells and assign `True` to all uncovered elements.
        """
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True

    def _step1(self):
        """
        Step 1

        Goal: Subtract the smallest element of each row from its elements.
            - All elements of the matrix are now non-negative.
            - Therefore, an assignment of total cost 0 is the minimum cost assignment.
            - This operation leads to at least one zero in each row.

        Procedure:
        - For each row of the matrix, find the smallest element and subtract it from every element in its row.
        - Go to Step 2.
        """
        self.cost_mat -= torch.min(self.cost_mat, dim=1)[0].unsqueeze(1)
        return 2

    def _step2(self):
        """
        Step 2

        Goal: Make sure assignment with cost sum 0 is feasible.

        Procedure:
        - Find a zero in the resulting cost matrix.
        - If there are no marked zeros in its row or column, mark the zero.
        - Repeat for each element in the matrix.
        - Go to step 3.
        """
        ind_out = torch.where(self.cost_mat == 0)
        ind, val = list(ind_out[0]), list(ind_out[1])
        for i, j in zip(ind, val):
            if self.col_uncovered[j] and self.row_uncovered[i]:
                self.marked[i, j] = 1
                self.col_uncovered[j] = False
                self.row_uncovered[i] = False

        self._reset_uncovered_mat()
        return 3

    def _step3(self) -> int:
        """
        Step 3

        Goal: All zeros in the matrix must be covered by marking with the least numbers of rows and columns.

        Procedure:
            - Cover each column containing a marked zero.
                - If n columns are covered, the marked zeros describe a complete set of unique assignments.
                In this case, Go to Step 0 (Done state)
                - Otherwise, Go to Step 4.
        """
        marked = self.marked == 1
        self.col_uncovered[torch.any(marked, dim=0)] = False
        if marked.sum() < self.cost_mat.shape[0]:
            return 4  # Go to step 4
        else:
            return 0  # Go to step 0 (Done state)


    def _step4(self, bypass: bool = False) -> int:
        """
        Step 4

        Goal: Cover all columns containing a marked zero.

        Procedure:
        - Find a non-covered zero and put a prime mark on it.
            - If there is no marked zero in the row containing this primed zero, Go to Step 5.
            - Otherwise, cover this row and uncover the column containing the marked zero.
        - Continue in this manner until there are no uncovered zeros left.
        - Save the smallest uncovered value.
        - Go to Step 6.
        """
        # We convert to int as numpy operations are faster on int
        cost_mat = (self.cost_mat == 0).int()
        covered_cost_mat = cost_mat * self.row_uncovered.unsqueeze(1)
        covered_cost_mat *= self.col_uncovered.long()
        row_len, col_len = self.cost_mat.shape
        if not bypass:
            while True:
                flat_idx = torch.argmax(covered_cost_mat).item()
                urv = unravel_index(flat_idx, torch.tensor([row_len, col_len]))
                row, col = int(urv[0].item()), int(urv[1].item())
                if covered_cost_mat[row, col] == 0:
                    return 6
                self.marked[row, col] = 2  # prime
                mark_col = torch.argmax((self.marked[row] == 1).int())
                if self.marked[row, mark_col] != 1:
                    self.zero_row = torch.tensor(row)
                    self.zero_col = torch.tensor(col)
                    return 5
                else:
                    col = int(mark_col.item())
                    self.row_uncovered[row] = False
                    self.col_uncovered[col] = True
                    covered_cost_mat[:, col] = cost_mat[:, col] * self.row_uncovered
                    covered_cost_mat[row] = 0
        return 0


    # def _step4(self, bypass: bool = False) -> int:
    #     """
    #     Step 4

    #     Goal: Cover all columns containing a marked zero.

    #     Procedure:
    #     - Find a non-covered zero and put a prime mark on it.
    #         - If there is no marked zero in the row containing this primed zero, Go to Step 5.
    #         - Otherwise, cover this row and uncover the column containing the marked zero.
    #     - Continue in this manner until there are no uncovered zeros left.
    #     - Save the smallest uncovered value.
    #     - Go to Step 6.
    #     """
    #     # We convert to int as numpy operations are faster on int
    #     cost_mat = (self.cost_mat == 0).int()
    #     covered_cost_mat = cost_mat * self.row_uncovered.unsqueeze(1)
    #     covered_cost_mat *= self.col_uncovered.long()
    #     row_len, col_len = self.cost_mat.shape
    #     if not bypass:
    #         while True:
    #             urv = unravel_index(
    #                 torch.argmax(covered_cost_mat).item(),
    #                 torch.tensor([col_len, row_len]),
    #             )
    #             row, col = int(urv[0].item()), int(urv[1].item())
    #             if covered_cost_mat[row, col] == 0:
    #                 return 6
    #             else:
    #                 self.marked[row, col] = (
    #                     2  # Find the first marked element in the row
    #                 )
    #                 mark_col = torch.argmax((self.marked[row] == 1).int())
    #                 if self.marked[row, mark_col] != 1:  # No marked element in the row
    #                     self.zero_row = torch.tensor(row)
    #                     self.zero_col = torch.tensor(col)
    #                     return 5
    #                 else:
    #                     col = mark_col
    #                     self.row_uncovered[row] = False
    #                     self.col_uncovered[col] = True
    #                     covered_cost_mat[:, col] = cost_mat[:, col] * self.row_uncovered
    #                     covered_cost_mat[row] = 0
    #     return 0

    def _step5(self) -> int:
        """
        Step 5

        Goal: Construct a series of alternating primed and marked zeros as follows.

        Procedure:
        - Let Z0 represent the uncovered primed zero found in Step 4.
        - Let Z1 denote the marked zero in the column of Z0 (if any).
        - Let Z2 denote the primed zero in the row of Z1 (there will always be one).
        - Continue until the series terminates at a primed zero that has no marked zero in its column.
        - Unmark each marked zero of the series.
        - Mark each primed zero of the series.
        - Erase all primes and uncover every line in the matrix.
        - Return to Step 3
        """
        count = torch.tensor(0)
        path = self.path
        path[count, 0] = self.zero_row.long()
        path[count, 1] = self.zero_col.long()

        while True:  # Unmark each marked zero of the series
            # Find the first marked element in the col defined by the path (= `val`)
            row = torch.argmax((self.marked[:, path[count, 1]] == 1).int())

            if self.marked[row, path[count, 1]] != 1:
                # Could not find one
                break
            else:
                count += 1
                path[count, 0] = row
                path[count, 1] = path[count - 1, 1]

            # Find the first prime element in the row defined by the first path step
            col = int(torch.argmax((self.marked[path[count, 0]] == 2).int()))
            if self.marked[row, col] != 2:
                col = -1
            count += 1
            path[count, 0] = path[count - 1, 0]
            path[count, 1] = col

        # Convert paths
        for i in range(int(count.item()) + 1):
            if self.marked[path[i, 0], path[i, 1]] == 1:
                self.marked[path[i, 0], path[i, 1]] = 0
            else:
                self.marked[path[i, 0], path[i, 1]] = 1

        self._reset_uncovered_mat()

        # Remove all prime markings in marked matrix
        self.marked[self.marked == 2] = 0
        return 3

    def _step6(self) -> int:
        """
        Step 6

        Goal: Prepare for another iteration by modifying the cost matrix.

        Procedure:
        - Add the value found in Step 4 to every element of each covered row.
        - Subtract it from every element of each uncovered column.
        - Return to Step 4 without altering any marks, primes, or covered lines.
        """
        if torch.any(self.row_uncovered) and torch.any(self.col_uncovered):
            row_minval = torch.min(self.cost_mat[self.row_uncovered], dim=0)[0]
            minval = torch.min(row_minval[self.col_uncovered])
            self.cost_mat[~self.row_uncovered] += minval
            self.cost_mat[:, self.col_uncovered] -= minval
        return 4


@torch.jit.script
def linear_sum_assignment(cost_matrix: torch.Tensor, max_size: int = 100):
    """
    Launch the linear sum assignment algorithm on a cost matrix.

    Args:
        cost_matrix (Tensor): The cost matrix of shape (N, M) where M should be larger than N.

    Returns:
        row_index (Tensor): The row indices of the optimal assignments.
        col_index (Tensor): The column indices of the optimal assignments.
    """
    cost_matrix = cost_matrix.clone().detach()


    if len(cost_matrix.shape) != 2:
        raise ValueError(f"2-d tensor is expected but got a {cost_matrix.shape} tensor")
    if max(cost_matrix.shape) > max_size:
        raise ValueError(
            f"Cost matrix size {cost_matrix.shape} is too large. The maximum supported size is {max_size}x{max_size}."
        )

    # The algorithm expects more columns than rows in the cost matrix.
    if cost_matrix.shape[1] < cost_matrix.shape[0]:
        cost_matrix = cost_matrix.T
        transposed = True
    else:
        transposed = False

    lap_solver = LinearSumAssignmentSolver(cost_matrix)
    f_int: int = 0 if 0 in cost_matrix.shape else 1
    # while step is not Done (step 0):
    # NOTE: torch.jit.scipt does not support getattr with string argument.
    # Do not use getattr(lap_solver, f"_step{f_int}")()
    while f_int != 0:
        if f_int == 1:
            f_int = lap_solver._step1()
        elif f_int == 2:
            f_int = lap_solver._step2()
        elif f_int == 3:
            f_int = lap_solver._step3()
        elif f_int == 4:
            f_int = lap_solver._step4()
        elif f_int == 5:
            f_int = lap_solver._step5()
        elif f_int == 6:
            f_int = lap_solver._step6()

    if transposed:
        marked = lap_solver.marked.T
    else:
        marked = lap_solver.marked
    row_index, col_index = torch.where(marked == 1)
    return row_index, col_index


def word_error_rate(
    hypotheses: List[str], references: List[str], use_cer=False
) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer (float): average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        # May deprecate using editdistance in future release for here and rest of codebase
        # once we confirm jiwer is reliable.
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float("inf")
    return wer


def calculate_session_cpWER_bruteforce(
    spk_hypothesis: List[str], spk_reference: List[str], limit_hypo_number: bool = False
) -> Tuple[float, str, str]:
    """
    Calculate cpWER with actual permutations in brute-force way when LSA algorithm cannot deliver the correct result.

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_reference = ["hi how are you well that's nice", "i'm good yeah how is your sister"]

    Returns:
        cpWER (float):
            cpWER value for the given session.
        min_perm_hyp_trans (str):
            Hypothesis transcript containing the permutation that minimizes WER. Words are separated by spaces.
        ref_trans (str):
            Reference transcript in an arbitrary permutation. Words are separated by spaces.
    """
    p_wer_list, permed_hyp_lists, perm_indices = [], [], []
    ref_word_list = []

    # Concatenate the hypothesis transcripts into a list
    for spk_id, word_list in enumerate(spk_reference):
        ref_word_list.append(word_list)
    ref_trans = " ".join(ref_word_list)

    # Calculate WER for every permutation
    for hyp_perm in permutations(range(len(spk_hypothesis))):
        if limit_hypo_number:
            hyp_perm = hyp_perm[: len(spk_reference)]

        hyp_trans = " ".join(spk_hypothesis[i] for i in hyp_perm)
        permed_hyp_lists.append(hyp_trans)
        perm_indices.append(hyp_perm)

        # Calculate a WER value of the permuted and concatenated transcripts
        p_wer = word_error_rate(hypotheses=[hyp_trans], references=[ref_trans])
        p_wer_list.append(p_wer)

    # Find the lowest WER and its hypothesis transcript
    argmin_idx = np.argmin(p_wer_list)
    min_perm_hyp_trans = permed_hyp_lists[argmin_idx]
    cpWER = p_wer_list[argmin_idx]
    best_perm = perm_indices[argmin_idx]
    return cpWER, min_perm_hyp_trans, ref_trans, best_perm


def calculate_session_cpWER(
    spk_hypothesis: List[str], spk_reference: List[str], use_lsa_only: bool = False, limit_hypo_number: bool = False
) -> Tuple[float, str, str]:
    """
    Calculate a session-level concatenated minimum-permutation word error rate (cpWER) value. cpWER is
    a scoring method that can evaluate speaker diarization and speech recognition performance at the same time.
    cpWER is calculated by going through the following steps.

    1. Concatenate all utterances of each speaker for both reference and hypothesis files.
    2. Compute the WER between the reference and all possible speaker permutations of the hypothesis.
    3. Pick the lowest WER among them (this is assumed to be the best permutation: `min_perm_hyp_trans`).

    cpWER was proposed in the following article:
        CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings
        https://arxiv.org/pdf/2004.09249.pdf

    Implementation:
        - Brute force permutation method for calculating cpWER has a time complexity of `O(n!)`.
        - To reduce the computational burden, linear sum assignment (LSA) algorithm is applied
          (also known as Hungarian algorithm) to find the permutation that leads to the lowest WER.
        - In this implementation, instead of calculating all WER values for all permutation of hypotheses,
          we only calculate WER values of (estimated number of speakers) x (reference number of speakers)
          combinations with `O(n^2)`) time complexity and then select the permutation that yields the lowest
          WER based on LSA algorithm.
        - LSA algorithm has `O(n^3)` time complexity in the worst case.
        - We cannot use LSA algorithm to find the best permutation when there are more hypothesis speakers
          than reference speakers. In this case, we use the brute-force permutation method instead.

          Example:
              >>> transcript_A = ['a', 'b', 'c', 'd', 'e', 'f'] # 6 speakers
              >>> transcript_B = ['a c b d', 'e f'] # 2 speakers

              [case1] hypothesis is transcript_A, reference is transcript_B
              [case2] hypothesis is transcript_B, reference is transcript_A

              LSA algorithm based cpWER is:
                [case1] 4/6 (4 deletion)
                [case2] 2/6 (2 substitution)
              brute force permutation based cpWER is:
                [case1] 0
                [case2] 2/6 (2 substitution)

    Args:
        spk_hypothesis (list):
            List containing the hypothesis transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_hypothesis = ["hey how are you we that's nice", "i'm good yes hi is your sister"]

        spk_reference (list):
            List containing the reference transcript for each speaker. A list containing the sequence
            of words is assigned for each speaker.

            Example:
            >>> spk_reference = ["hi how are you well that's nice", "i'm good yeah how is your sister"]

    Returns:
        cpWER (float):
            cpWER value for the given session.
        min_perm_hyp_trans (str):
            Hypothesis transcript containing the permutation that minimizes WER. Words are separated by spaces.
        ref_trans (str):
            Reference transcript in an arbitrary permutation. Words are separated by spaces.
    """
    # Get all pairs of (estimated num of spks) x (reference num of spks) combinations
    hyp_ref_pair = [spk_hypothesis, spk_reference]
    all_pairs = list(itertools.product(*hyp_ref_pair))

    num_hyp_spks, num_ref_spks = len(spk_hypothesis), len(spk_reference)

    if not use_lsa_only and num_ref_spks < num_hyp_spks:
        # Brute force algorithm when there are more speakers in the hypothesis
        cpWER, min_perm_hyp_trans, ref_trans, best_perm = calculate_session_cpWER_bruteforce(
            spk_hypothesis, spk_reference, limit_hypo_number = limit_hypo_number
        )
    else:
        # Calculate WER for each speaker in hypothesis with reference
        # There are (number of hyp speakers) x (number of ref speakers) combinations
        lsa_wer_list = []
        for spk_hyp_trans, spk_ref_trans in all_pairs:
            spk_wer = word_error_rate(
                hypotheses=[spk_hyp_trans], references=[spk_ref_trans]
            )
            lsa_wer_list.append(spk_wer)

        # Make a cost matrix and calculate a linear sum assignment on the cost matrix.
        # Row is hypothesis index and column is reference index
        cost_wer = torch.tensor(lsa_wer_list).reshape(
            [len(spk_hypothesis), len(spk_reference)]
        )
        row_hyp_ind, col_ref_ind = linear_sum_assignment(cost_wer)

        # best_perm[i] = hyp speaker index assigned to ref speaker i
        best_perm = tuple(np.argsort(col_ref_ind).tolist())
        hyp_permed = [spk_hypothesis[k] for k in best_perm]
        min_perm_hyp_trans = " ".join(hyp_permed)

        # Concatenate the reference transcripts into a string variable
        ref_trans = " ".join(spk_reference)

        # Calculate a WER value from the permutation that yields the lowest WER.
        cpWER = word_error_rate(hypotheses=[min_perm_hyp_trans], references=[ref_trans])

    return cpWER, min_perm_hyp_trans, ref_trans, best_perm


if __name__ == "__main__":
    ref = ["hi how are you well that's nice", "i'm good yeah how is your sister"]
    hyps = ["hey how are you we that's nice", "i'm good yes hi is your sister"]
    print(calculate_session_cpWER(hyps, ref))
