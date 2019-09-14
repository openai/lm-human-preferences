import tensorflow as tf

from lm_human_preferences.language import model
from lm_human_preferences.utils import core as utils


def sample_sequence(*, step, model_hparams, length, batch_size=None, context=None,
                    temperature=1, top_k=0, top_p=1.0, extra_outputs={}, cond=None):
    """
    Sampling from an autoregressive sequence model.

    Inputs:
        step: A function which takes model hparams, a tokens Tensor, past, and
            returns a dictionary with 'logits' and 'presents', and any extra vars.
        context: Includes start tokens.
        extra_outputs: Map from extra output key to dtype
    Returns:
        A dict with keys 'presents', 'logits', and any keys in extra_outputs
    """

    with tf.name_scope('sample_seq'):
        batch_size, *_ = utils.shape_list(context)

        beta = 1 / tf.maximum(tf.cast(temperature, tf.float32), 1e-10)

        context_output = step(model_hparams, context)
        logits = tf.cast(context_output['logits'][:,-1], tf.float32)

        first_output_logits = tf.cast(beta, logits.dtype) * logits
        first_outputs = utils.sample_from_logits(first_output_logits)
        first_logprobs = utils.logprobs_from_logits(logits=first_output_logits, labels=first_outputs)

        def body(past, prev, output, logprobs, *extras):
            next_outputs = step(model_hparams, prev[:, tf.newaxis], past=past,
                                past_tokens=output[:, :-1])
            logits = tf.cast(next_outputs['logits'], tf.float32) * beta
            if top_k != 0:
                logits = tf.cond(tf.equal(top_k, 0),
                                 lambda: logits,
                                 lambda: utils.take_top_k_logits(logits, top_k))
            if top_p != 1.0:
                logits = utils.take_top_p_logits(logits, top_p)
            next_sample = utils.sample_from_logits(logits, dtype=tf.int32)

            next_logprob = utils.logprobs_from_logits(logits=logits, labels=next_sample)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(next_sample, axis=[1]),
                tf.concat([output, next_sample], axis=1),
                tf.concat([logprobs, next_logprob], axis=1),
                *[tf.concat([prev, next_outputs[k]], axis=1) for k, prev in zip(extra_outputs, extras)],
            ]

        try:
            shape_batch_size = int(batch_size)
        except TypeError:
            shape_batch_size = None
        if cond is None:
            def always_true(*args):
                return True
            cond = always_true
        presents, _, tokens, logprobs, *extras = tf.while_loop(
            body=body,
            cond=cond,
            loop_vars=[
                context_output['presents'], # past
                first_outputs, # prev
                tf.concat([context, first_outputs[:, tf.newaxis]], axis=1), # output
                first_logprobs[:, tf.newaxis], #logprobs
                *[context_output[k][:, -1:] for k in extra_outputs] # extras
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=model_hparams, batch_size=shape_batch_size)),
                tf.TensorShape([shape_batch_size]),
                tf.TensorShape([shape_batch_size, None]),
                tf.TensorShape([shape_batch_size, None]),
                *[tf.TensorShape([shape_batch_size, None]) for _ in extra_outputs]
            ],
            maximum_iterations=length-1,
            back_prop=False,
            parallel_iterations=2,
        )

        return dict(tokens=tokens, presents=presents, logprobs=logprobs, **dict(zip(extra_outputs, extras)))
