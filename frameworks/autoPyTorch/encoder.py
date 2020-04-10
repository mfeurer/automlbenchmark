"""
**datautils** module provide some utility functions for data manipulation.

important
    This is (and should remain) the only non-framework module with dependencies to libraries like pandas or sklearn
    until replacement by simpler/lightweight versions to avoid potential version conflicts with libraries imported by benchmark frameworks.
    Also, this module is intended to be imported by frameworks integration modules,
    therefore, it should have no dependency to any other **amlb** module outside **utils**.
"""
import logging
import os

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, roc_auc_score  # just aliasing
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder    # from sklearn 0.20

class Encoder(TransformerMixin):
    """
    Overly complex "generic" encoder that can handle missing values, auto encoded format (e.g. int for target, float for predictors)...
    Should never have written this, but does the job currently. However, should think about simpler single-purpose approach.
    """

    def __init__(self, type='label', target=True, encoded_type=int,
                 missing_policy='ignore', missing_values=None, missing_replaced_by=''):
        """
        :param type:
        :param target:
        :param missing_policy: one of ['ignore', 'mask', 'encode'].
            ignore: use only if there's no missing value for sure data to be transformed, otherwise it may raise an error during transform()
            mask: replace missing values only internally
            encode: encode all missing values as the encoded value of missing_replaced_by
        :param missing_values:
        :param missing_replaced_by:
        """
        super().__init__()
        assert missing_policy in ['ignore', 'mask', 'encode']
        self.for_target = target
        self.missing_policy = missing_policy
        self.missing_values = set(missing_values).union([None]) if missing_values else {None}
        self.missing_replaced_by = missing_replaced_by
        self.missing_encoded_value = None
        self.str_encoder = None
        self.classes = None
        self._enc_classes_ = None
        # self.encoded_type = int if target else encoded_type
        self.encoded_type = encoded_type
        if type == 'label':
            self.delegate = LabelEncoder() if target else OrdinalEncoder()
        elif type == 'one-hot':
            self.str_encoder = None if target else LabelEncoder()
            self.delegate = LabelBinarizer() if target else OneHotEncoder(sparse=False, handle_unknown='ignore')
        elif type == 'no-op':
            self.delegate = None
            # self.encoded_type = encoded_type
        else:
            raise ValueError("Encoder `type` should be one of {}.".format(['label', 'one-hot']))

    @property
    def _ignore_missing(self):
        return self.for_target or self.missing_policy == 'ignore'

    @property
    def _mask_missing(self):
        return not self.for_target and self.missing_policy == 'mask'

    @property
    def _encode_missing(self):
        return not self.for_target and self.missing_policy == 'encode'

    def _reshape(self, vec):
        return vec if self.for_target else vec.reshape(-1, 1)

    def fit(self, vec):
        """

        :param vec: must be a line vector (array)
        :return:
        """
        if not self.delegate:
            return self

        vec = np.asarray(vec, dtype=object)
        self.classes = np.unique(vec) if self._ignore_missing else np.unique(np.insert(vec, 0, self.missing_replaced_by))
        self._enc_classes_ = self.str_encoder.fit_transform(self.classes) if self.str_encoder else self.classes

        if self._mask_missing:
            self.missing_encoded_value = self.delegate.fit_transform(self._reshape(self.classes))[0]
        else:
            self.delegate.fit(self._reshape(self.classes))
        return self

    def transform(self, vec, **params):
        """

        :param vec: must be single value (str) or a line vector (array)
        :param params:
        :return:
        """
        return_value = lambda v: v
        if isinstance(vec, str):
            vec = [vec]
            return_value = lambda v: v[0]

        vec = np.asarray(vec, dtype=object)

        if not self.delegate:
            return return_value(vec.astype(self.encoded_type, copy=False))

        if self.str_encoder:
            vec = self.str_encoder.transform(vec)

        if self._mask_missing or self._encode_missing:
            mask = [v in self.missing_values for v in vec]
            if any(mask):
                # if self._mask_missing:
                #     missing = vec[mask]
                vec[mask] = self.missing_replaced_by
                res = self.delegate.transform(self._reshape(vec), **params).astype(self.encoded_type, copy=False)
                if self._mask_missing:
                    res[mask] = np.NaN if self.encoded_type == float else None
                return return_value(res)

        return return_value(self.delegate.transform(self._reshape(vec), **params).astype(self.encoded_type, copy=False))

    def inverse_transform(self, vec, **params):
        """

        :param vec: must a single value or line vector (array)
        :param params:
        :return:
        """
        if not self.delegate:
            return vec

        # TODO: handle mask
        vec = np.asarray(vec, dtype=object).astype(self.encoded_type, copy=False)
        return self.delegate.inverse_transform(vec, **params)

