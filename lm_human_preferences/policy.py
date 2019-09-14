import tensorflow as tf

from lm_human_preferences.language import model, sample
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils.core import Schema


class Policy:
    def __init__(
            self,
            trained_model, *,
            scope=None, use_resource=False,
            embed_queries=lambda queries: queries,
            temperature=1.0, is_root=True,
            build_respond=True,
    ):
        self.trained_model = trained_model
        self.model_hparams = trained_model.hparams()
        self.is_root = is_root

        self.use_resource = use_resource
        self.encoder = self.trained_model.encoding.get_encoder()

        with tf.variable_scope(scope, 'transformer_policy', use_resource=self.use_resource) as s:
            self.scope = s
            self.model = model.Model(
                hparams=self.model_hparams,
                scalar_heads=['value'])

        self.built = False
        self.embed_queries = embed_queries
        self.temperature = temperature
        self.padding_token = self.encoder.padding_token

        if build_respond:
            self.respond = utils.graph_function(
                queries=Schema(tf.int32, (None, None)),
                length=Schema(tf.int32, ()),
            )(self.respond_op)
        self.analyze_responses = utils.graph_function(
            queries=Schema(tf.int32, (None, None)),
            responses=Schema(tf.int32, (None, None)),
        )(self.analyze_responses_op)

    def get_encoder(self):
        return self.encoder

    def step_core(self, model_hparams, tokens, past=None, past_tokens=None, do_dropout=False, name=None):
        with tf.name_scope(name, 'step'):
            with tf.variable_scope(
                    self.scope,
                    reuse=self.built,
                    auxiliary_name_scope=not self.built,
                    use_resource=self.use_resource):
                lm_output = self.model(X=tokens, past=past, past_tokens=past_tokens,
                                       do_dropout=do_dropout, padding_token=self.padding_token)

                # need to slice logits since we don't want to generate special tokens
                logits = lm_output['lm_logits'][:,:,:self.model_hparams.n_vocab]
                presents = lm_output['present']
                value = lm_output['value']
                if not self.built:
                    self._set_initializers()
                self.built = True
                return {
                    'logits': logits,
                    'values': value,
                    'presents': presents,
                }

    def ensure_built(self):
        if not self.built:
            with tf.name_scope('dummy'):
                self.step_core(self.model_hparams, tokens=tf.zeros([0,0], dtype=tf.int32))

    def get_params(self):
        self.ensure_built()
        params = utils.find_trainable_variables(self.scope.name)
        assert len(params) > 0
        return params

    def _set_initializers(self):
        """Change initializers to load a language model from a tensorflow checkpoint."""
        # Skip if
        # 1. We're not rank 0.  Values will be copied from there.
        # 2. We want random initialization.  Normal initialization will do the work.
        if not self.is_root or self.trained_model.name == 'test':
            return

        with tf.init_scope():
            scope = self.scope.name

            # Initialize!
            params = {v.op.name: v for v in utils.find_trainable_variables(scope)}
            self.trained_model.init_op(params, new_scope=scope)

    def respond_op(self, queries, length):
        contexts = self.embed_queries(queries)
        context_length = tf.shape(contexts)[1]
        result = sample.sample_sequence(
            step=self.step_core,
            context=contexts,
            length=length,
            model_hparams=self.model_hparams,
            temperature=self.temperature,
            extra_outputs={'values':tf.float32},
        )
        return dict(
            responses=result['tokens'][:, context_length:],
            logprobs=result['logprobs'],
            values=result['values'],
        )

    def analyze_responses_op(self, queries, responses):
        contexts = self.embed_queries(queries)
        context_length = tf.shape(contexts)[1]
        tokens = tf.concat([contexts, responses], axis=1)
        result = self.step_core(self.model_hparams, tokens)
        logits = result['logits'][:, context_length-1:-1]

        logits /= self.temperature
        return dict(
            logprobs = utils.logprobs_from_logits(logits=logits, labels=responses),
            entropies = utils.entropy_from_logits(logits),
            values = result['values'][:, context_length-1:-1],
        )

