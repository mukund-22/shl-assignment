# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from inspect import signature
from itertools import chain, combinations
from math import ceil, floor

import numpy as np
from scipy.special import comb

# Change from relative import to absolute import
from utils import (
    _safe_indexing,
    check_random_state,
    indexable,
    metadata_routing,
)

from utils._array_api import (
    _convert_to_numpy,
    ensure_common_namespace_device,
    get_namespace,
)

from utils._param_validation import Interval, RealNotInt, validate_params
from utils.extmath import _approximate_mode
from utils.metadata_routing import _MetadataRequester
from utils.multiclass import type_of_target
from utils.validation import _num_samples, check_array, column_or_1d

__all__ = [
    "BaseCrossValidator",
    "KFold",
    "GroupKFold",
    "LeaveOneGroupOut",
    "LeaveOneOut",
    "LeavePGroupsOut",
    "LeavePOut",
    "RepeatedStratifiedKFold",
    "RepeatedKFold",
    "ShuffleSplit",
    "GroupShuffleSplit",
    "StratifiedKFold",
    "StratifiedGroupKFold",
    "StratifiedShuffleSplit",
    "PredefinedSplit",
    "train_test_split",
    "check_cv",
]


class _UnsupportedGroupCVMixin:
    """Mixin for splitters that do not support Groups."""

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

     
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        return super().split(X, y, groups=groups)


class GroupsConsumerMixin(_MetadataRequester):
    """A Mixin to ``groups`` by default.

    This Mixin makes the object to request ``groups`` by default as ``True``.

    .. versionadded:: 1.3
    """

    __metadata_request__split = {"groups": True}


class BaseCrossValidator(_MetadataRequester, metaclass=ABCMeta):
    """Base class for all cross-validators.

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""

    def __repr__(self):
        return _build_repr(self)


class LeaveOneOut(_UnsupportedGroupCVMixin, BaseCrossValidator):
 

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        if n_samples <= 1:
            raise ValueError(
                "Cannot perform LeaveOneOut with n_samples={}.".format(n_samples)
            )
        return range(n_samples)

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return _num_samples(X)


class LeavePOut(_UnsupportedGroupCVMixin, BaseCrossValidator):
    """Leave-P-Out cross-validator.

    Provides train/test indices to split data in train/test sets. This results
    in testing on all distinct samples of size p, while the remaining n - p
    samples form the training set in each iteration.

  
    """

    def __init__(self, p):
        self.p = p

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        if n_samples <= self.p:
            raise ValueError(
                "p={} must be strictly less than the number of samples={}".format(
                    self.p, n_samples
                )
            )
        for combination in combinations(range(n_samples), self.p):
            yield np.array(combination)

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return int(comb(_num_samples(X), self.p, exact=True))


class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for K-Fold cross-validators and TimeSeriesSplit."""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                (
                    "Setting a random_state has no effect since shuffle is "
                    "False. You should leave "
                    "random_state to its default (None), or set shuffle=True."
                ),
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class KFold(_UnsupportedGroupCVMixin, _BaseKFold):

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class GroupKFold(GroupsConsumerMixin, _BaseKFold):
    """K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)

        unique_groups, group_idx = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        if self.shuffle:
            # Split and shuffle unique groups across n_splits
            rng = check_random_state(self.random_state)
            unique_groups = rng.permutation(unique_groups)
            split_groups = np.array_split(unique_groups, self.n_splits)

            for test_group_ids in split_groups:
                test_mask = np.isin(groups, test_group_ids)
                yield np.where(test_mask)[0]

        else:
            # Weight groups by their number of occurrences
            n_samples_per_group = np.bincount(group_idx)

            # Distribute the most frequent groups first
            indices = np.argsort(n_samples_per_group)[::-1]
            n_samples_per_group = n_samples_per_group[indices]

            # Total weight of each fold
            n_samples_per_fold = np.zeros(self.n_splits)

            # Mapping from group index to fold index
            group_to_fold = np.zeros(len(unique_groups))

            # Distribute samples by adding the largest weight to the lightest fold
            for group_index, weight in enumerate(n_samples_per_group):
                lightest_fold = np.argmin(n_samples_per_fold)
                n_samples_per_fold[lightest_fold] += weight
                group_to_fold[indices[group_index]] = lightest_fold

            indices = group_to_fold[group_idx]

            for f in range(self.n_splits):
                yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        
        return super().split(X, y, groups)


class StratifiedKFold(_BaseKFold):
    #Stratified K-Fold cross-validator.


    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        # XXX: as of now, cross-validation splitters only operate in NumPy-land
        # without attempting to leverage array API namespace features. However
        # they might be fed by array API inputs, e.g. in CV-enabled estimators so
        # we need the following explicit conversion:
        xp, is_array_api = get_namespace(y)
        if is_array_api:
            y = _convert_to_numpy(y, xp)
        else:
            y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype="i")
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set. """
        
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


class StratifiedGroupKFold(GroupsConsumerMixin, _BaseKFold):


    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        # Implementation is based on this kaggle kernel:
       
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )

        y = column_or_1d(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        n_smallest_class = np.min(y_cnt)
        if self.n_splits > n_smallest_class:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (n_smallest_class, self.n_splits),
                UserWarning,
            )
        n_classes = len(y_cnt)

        _, groups_inv, groups_cnt = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1

        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)

        if self.shuffle:
            rng.shuffle(y_counts_per_group)

        # Stable sort to keep shuffled order for groups with the same
        # class distribution variance
        sorted_groups_idx = np.argsort(
            -np.std(y_counts_per_group, axis=1), kind="mergesort"
        )

        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = self._find_best_fold(
                y_counts_per_fold=y_counts_per_fold,
                y_cnt=y_cnt,
                group_y_counts=group_y_counts,
            )
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)

        for i in range(self.n_splits):
            test_indices = [
                idx
                for idx, group_idx in enumerate(groups_inv)
                if group_idx in groups_per_fold[i]
            ]
            yield test_indices

    def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # Summarise the distribution over classes in each proposed fold
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = (
                fold_eval < min_eval
                or np.isclose(fold_eval, min_eval)
                and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold


class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

   
    """

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        return self._split(X)

    def _split(self, X):
        """Generate indices to split data into training and test set."""
        
        (X,) = indexable(X)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )

        # Makes sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )

        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                yield (
                    indices[train_end - self.max_train_size : train_end],
                    indices[test_start : test_start + test_size],
                )
            else:
                yield (
                    indices[:train_end],
                    indices[test_start : test_start + test_size],
                )


class LeaveOneGroupOut(GroupsConsumerMixin, BaseCrossValidator):
    """Leave One Group Out cross-validator.

    Provides train/test indices to split data such that each training set is
    comprised of all samples except ones belonging to one specific group.
    Arbitrary domain specific group information is provided as an array of integers
    that encodes the group of each sample.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.
    """

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side-effects during iteration
        groups = check_array(
            groups, input_name="groups", copy=True, ensure_2d=False, dtype=None
        )
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups
            )
        for i in unique_groups:
            yield groups == i

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        return len(np.unique(groups))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        """
        return super().split(X, y, groups)


class LeavePGroupsOut(GroupsConsumerMixin, BaseCrossValidator):
    """Leave P Group(s) Out cross-validator.
    """

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(
            groups, input_name="groups", copy=True, ensure_2d=False, dtype=None
        )
        unique_groups = np.unique(groups)
        if self.n_groups >= len(unique_groups):
            raise ValueError(
                "The groups parameter contains fewer than (or equal to) "
                "n_groups (%d) numbers of unique groups (%s). LeavePGroupsOut "
                "expects that at least n_groups + 1 (%d) unique groups be "
                "present" % (self.n_groups, unique_groups, self.n_groups + 1)
            )
        combi = combinations(range(len(unique_groups)), self.n_groups)
        for indices in combi:
            test_index = np.zeros(_num_samples(X), dtype=bool)
            for l in unique_groups[np.array(indices)]:
                test_index[groups == l] = True
            yield test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        return int(comb(len(np.unique(groups)), self.n_groups, exact=True))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        """
        return super().split(X, y, groups)


class _RepeatedSplits(_MetadataRequester, metaclass=ABCMeta):
    """Repeated splits for an arbitrary randomized CV splitter.

   
    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def __init__(self, cv, *, n_repeats=10, random_state=None, **cvargs):
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError("Number of repetitions must be of Integral type.")

        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")

        if any(key in cvargs for key in ("random_state", "shuffle")):
            raise ValueError("cvargs must not contain random_state or shuffle.")

        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cvargs = cvargs

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.

        
        """
        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)

        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        """
        rng = check_random_state(self.random_state)
        cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
        return cv.get_n_splits(X, y, groups) * self.n_repeats

    def __repr__(self):
        return _build_repr(self)


class RepeatedKFold(_UnsupportedGroupCVMixin, _RepeatedSplits):
    """Repeated K-Fold cross validator.
.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            KFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits
        )


class RepeatedStratifiedKFold(_UnsupportedGroupCVMixin, _RepeatedSplits):
    """Repeated Stratified K-Fold cross validator.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            StratifiedKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        """
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups=groups)


class BaseShuffleSplit(_MetadataRequester, metaclass=ABCMeta):
    """Base class for *ShuffleSplit.

    """

    # This indicates that by default CV splitters don't have a "groups" kwarg,
    # unless indicated by inheriting from ``GroupsConsumerMixin``.
    # This also prevents ``set_split_request`` to be generated for splitters
    # which don't support ``groups``.
    __metadata_request__split = {"groups": metadata_routing.UNUSED}

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        """
        X, y, groups = indexable(X, y, groups)
        for train, test in self._iter_indices(X, y, groups):
            yield train, test

    def _iter_indices(self, X, y=None, groups=None):
        """Generate (train, test) indices"""
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            # random partition
            permutation = rng.permutation(n_samples)
            ind_test = permutation[:n_test]
            ind_train = permutation[n_test : (n_test + n_train)]
            yield ind_train, ind_test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

     
        """
        return self.n_splits

    def __repr__(self):
        return _build_repr(self)


class ShuffleSplit(_UnsupportedGroupCVMixin, BaseShuffleSplit):
    """Random permutation cross-validator.

    """

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.1


class GroupShuffleSplit(GroupsConsumerMixin, BaseShuffleSplit):
    """Shuffle-Group(s)-Out cross-validation iterator.

    """

    def __init__(
        self, n_splits=5, *, test_size=None, train_size=None, random_state=None
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.2

    def _iter_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name="groups", ensure_2d=False, dtype=None)
        classes, group_indices = np.unique(groups, return_inverse=True)
        for group_train, group_test in super()._iter_indices(X=classes):
            # these are the indices of classes in the partition
            # invert them into data indices

            train = np.flatnonzero(np.isin(group_indices, group_train))
            test = np.flatnonzero(np.isin(group_indices, group_test))

            yield train, test

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

    """

    def __init__(
        self, n_splits=10, *, test_size=None, train_size=None, random_state=None
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.1

    def _iter_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        # Convert to numpy as not all operations are supported by the Array API.
        # `y` is probably never a very large array, which means that converting it
        # should be cheap
        xp, _ = get_namespace(y)
        y = _convert_to_numpy(y, xp=xp)

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        if np.min(class_counts) < 2:
            raise ValueError(
                "The least populated class in y has only 1"
                " member, which is too few. The minimum"
                " number of groups for any class cannot"
                " be less than 2."
            )

        if n_train < n_classes:
            raise ValueError(
                "The train_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_train, n_classes)
            )
        if n_test < n_classes:
            raise ValueError(
                "The test_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_test, n_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)

            train = []
            test = []

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

                train.extend(perm_indices_class_i[: n_i[i]])
                test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


def _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=None):
    """
    Validation helper to check if the train/test sizes are meaningful w.r.t. the
    size of the data (n_samples).
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (
        test_size_type == "i"
        and (test_size >= n_samples or test_size <= 0)
        or test_size_type == "f"
        and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            "test_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(test_size, n_samples)
        )

    if (
        train_size_type == "i"
        and (train_size >= n_samples or train_size <= 0)
        or train_size_type == "f"
        and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            "train_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(train_size, n_samples)
        )

    if train_size is not None and train_size_type not in ("i", "f"):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    if train_size_type == "f" and test_size_type == "f" and train_size + test_size > 1:
        raise ValueError(
            "The sum of test_size and train_size = {}, should be in the (0, 1)"
            " range. Reduce test_size and/or train_size.".format(train_size + test_size)
        )

    if test_size_type == "f":
        n_test = ceil(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    if train_size_type == "f":
        n_train = floor(train_size * n_samples)
    elif train_size_type == "i":
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError(
            "The sum of train_size and test_size = %d, "
            "should be smaller than the number of "
            "samples %d. Reduce test_size and/or "
            "train_size." % (n_train + n_test, n_samples)
        )

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, test_size={} and train_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, test_size, train_size)
        )

    return n_train, n_test


class PredefinedSplit(BaseCrossValidator):
    """Predefined split cross-validator.

    """

    def __init__(self, test_fold):
        self.test_fold = np.array(test_fold, dtype=int)
        self.test_fold = column_or_1d(self.test_fold)
        self.unique_folds = np.unique(self.test_fold)
        self.unique_folds = self.unique_folds[self.unique_folds != -1]

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.

        """
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {self.__class__.__name__}",
                UserWarning,
            )
        return self._split()

    def _split(self):
        """Generate indices to split data into training and test set..
        """
        ind = np.arange(len(self.test_fold))
        for test_index in self._iter_test_masks():
            train_index = ind[np.logical_not(test_index)]
            test_index = ind[test_index]
            yield train_index, test_index

    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets."""
        for f in self.unique_folds:
            test_index = np.where(self.test_fold == f)[0]
            test_mask = np.zeros(len(self.test_fold), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.unique_folds)


class _CVIterableWrapper(BaseCrossValidator):
    """Wrapper class for old style cv objects and iterables."""

    def __init__(self, cv):
        self.cv = list(cv)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.cv)

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set
        """
        for train, test in self.cv:
            yield train, test


def check_cv(cv=5, y=None, *, classifier=False):
    """Input checker utility for building a cross-validator."""
    
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        if (
            classifier
            and (y is not None)
            and (type_of_target(y, input_name="y") in ("binary", "multiclass"))
        ):
            return StratifiedKFold(cv)
        else:
            return KFold(cv)

    if not hasattr(cv, "split") or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError(
                "Expected cv as an integer, cross-validation "
                "object (from sklearn.model_selection) "
                "or an iterable. Got %s." % cv
            )
        return _CVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification


@validate_params(
    {
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "random_state": ["random_state"],
        "shuffle": ["boolean"],
        "stratify": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """Split arrays or matrices into random train and test subsets."""
    
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=stratify))

    train, test = ensure_common_namespace_device(arrays[0], train, test)

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


# Tell nose that train_test_split is not a test.
# (Needed for external libraries that may use nose.)
# Use setattr to avoid mypy errors when monkeypatching.
setattr(train_test_split, "__test__", False)


def _pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    """
    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ",\n" + (1 + offset // 2) * " "
    for i, (k, v) in enumerate(sorted(params.items())):
        if isinstance(v, float):
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = "%s=%s" % (k, str(v))
        else:
            # use repr of the rest
            this_repr = "%s=%s" % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + "..." + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or "\n" in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(", ")
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = "".join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = "\n".join(l.rstrip(" ") for l in lines.split("\n"))
    return lines


def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted(
            [
                p.name
                for p in init_signature.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
        )
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", FutureWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, "cvargs"):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category is FutureWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return "%s(%s)" % (class_name, _pprint(params, offset=len(class_name)))


def _yields_constant_splits(cv):
    # Return True if calling cv.split() always returns the same splits
    # We assume that if a cv doesn't have a shuffle parameter, it shuffles by
    # default (e.g. ShuffleSplit). If it actually doesn't shuffle (e.g.
    # LeaveOneOut), then it won't have a random_state parameter anyway, in
    # which case it will default to 0, leading to output=True
    shuffle = getattr(cv, "shuffle", True)
    random_state = getattr(cv, "random_state", 0)
    return isinstance(random_state, numbers.Integral) or not shuffle
