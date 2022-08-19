"""Hierarchical Transformer implementation

The hierarchical Transformer architecture was introduced in https://arxiv.org/abs/2006.09265
"""
import tensorflow as tf

from ..layers import positional_encoding as pe
from . import transformer
from .beam_search import BeamSearch, flatten_beam_dim


def create_model(params, training, attn_weights=False):
    """
    Args:
        params: dict, hyperparameter dictionary
        training: bool, whether model is called in training mode or not
        attn_weights: bool, whether attention weights are part of the output, Defaults to False

    Raises:
        NotImplementedError: if we don't support a positional encoding
        NotImplementedError: if batch size is not same in all batches

    Returns:
        A Keras Model
    """

    params["return_attn_weights"] = attn_weights
    hierarchical_transformer = SeparatedHierarchicalTransformer(params)

    params["return_attn_weights"] = attn_weights
    input = tf.keras.layers.Input((None, None, None), dtype=tf.int32, name="input", ragged=True)
    transformer_inputs = {"input": input}
    model_inputs = [input]

    positional_encoding = tf.keras.layers.Input(
        (None, None, None, None), dtype=tf.float32, name="positional_encoding", ragged=True
    )
    transformer_inputs["positional_encoding"] = positional_encoding
    model_inputs.append(positional_encoding)

    if training:
        target = tf.keras.Input((None,), dtype=tf.int32, name="target")
        transformer_inputs["target"] = target
        model_inputs.append(target)

        # do not provide training argument so keras fit method can set it
        predictions, _ = hierarchical_transformer(transformer_inputs)
        predictions = transformer.TransformerMetricsLayer(params)([predictions, target])
        model = tf.keras.Model(inputs=model_inputs, outputs=predictions)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        mask = tf.cast(
            tf.math.logical_not(tf.math.equal(target, params["target_pad_id"])),
            params["dtype_float"],
        )
        loss = tf.keras.layers.Lambda(lambda x: loss_object(x[0], x[1], x[2]))(
            (target, predictions, mask)
        )
        model.add_loss(loss)
        return model
    else:
        # do not provide training argument so keras fit method can set it
        results = hierarchical_transformer(transformer_inputs)
        if attn_weights:
            outputs, scores, enc_attn_weights_local, enc_attn_weights_global, dec_attn_weights = (
                results["outputs"],
                results["scores"],
                results["enc_attn_weights_local"],
                results["enc_attn_weights_global"],
                results["dec_attn_weights"],
            )
            return tf.keras.Model(
                model_inputs,
                [
                    outputs,
                    scores,
                    enc_attn_weights_local,
                    enc_attn_weights_global,
                    dec_attn_weights,
                ],
            )
        else:
            outputs, scores = results["outputs"], results["scores"]
            return tf.keras.Model(model_inputs, [outputs, scores])


class SeparatedHierarchicalTransformer(tf.keras.Model):
    def __init__(self, params):
        """
        Args:
            params: hyperparameter dictionary containing the following keys:

                params_sep_local: list, A list of dicts with the following hyperparameters for each separated local encoder
                    input_vocab_size: int, size of input vocabulary
                    d_ff_enc: int, hidden dimension of local encoder feed-forward networks
                    ff_activation_enc: string, activation function used in local encoder feed-forward networks
                    num_heads_enc: int, number of local encoder attention heads
                    num_layers_enc: int, number of local encoder layers
                    input_dimensions: (int, int), dimensions of 2 dimensional input (1 sample).

                d_embed_enc: int, dimension of encoder embedding
                d_ff_enc: int, hidden dimension of local encoder feed-forward networks
                dropout_enc: float, percentage of dropped out encoder units
                ff_activation_enc: string, activation function used in dimension 0 encoder (global) feed-forward networks
                num_heads_enc: int, number of global encoder attention heads
                num_layers_enc: int, number of global encoder layer
                input_pad_id: int, encodes the padding token for inputs

                alpha: float, strength of normalization in beam search algorithm
                beam_size: int, number of beams kept by beam search algorithm
                d_embed_dec: int, dimension of decoder embedding
                d_ff_dec: int, hidden dimension of decoder feed-forward networks
                dropout_dec: float, percentage of dropped out decoder units
                dtype_float: tf.dtypes.Dtype(), datatype for floating point computations
                dtype_int: tf.dtypes.Dtype(), datatype for integer computations
                ff_activation_dec: string, activation function used in decoder feed-forward networks
                max_decode_length: int, maximum length of target sequence
                num_heads_dec: int, number of decoder attention heads
                num_layers_dec: int, number of decoder layer
                target_eos_id: int, encodes the end of string token for targets
                target_pad_id: int, encodes the padding token for targets
                target_start_id: int, encodes the start token for targets
        """
        super().__init__()
        self.params = params

        # create embedding layers, dropout layers and local transformer layers for each separated input
        self.encoder_embeddings = []
        self.encoder_dropouts = []
        self.encoders_stack_local = []
        for params_sep_local in params["params_sep_local"]:
            # create embedding layer
            self.encoder_embeddings.append(
                tf.keras.layers.Embedding(
                    params_sep_local["input_vocab_size"], params["d_embed_enc"]
                )
            )
            # create dropout layer
            self.encoder_dropouts.append(tf.keras.layers.Dropout(params["dropout_enc"]))
            # create local encoder layer
            self.encoders_stack_local.append(
                transformer.TransformerEncoder(
                    {
                        "d_embed_enc": params["d_embed_enc"],
                        "d_ff": params_sep_local["d_ff_enc"],
                        "dropout": params["dropout_enc"],
                        "ff_activation": params_sep_local["ff_activation_enc"],
                        "num_heads": params_sep_local["num_heads_enc"],
                        "num_layers_enc": params_sep_local["num_layers_enc"],
                    }
                )
            )
        # create global encoder layer
        self.encoder_stack_global = transformer.TransformerEncoder(
            {
                "d_embed_enc": params["d_embed_enc"],
                "d_ff": params["d_ff_enc"],
                "dropout": params["dropout_enc"],
                "ff_activation": params["ff_activation_enc"],
                "num_heads": params["num_heads_enc"],
                "num_layers_enc": params["num_layers_enc"],
            }
        )

        self.decoder_embedding = tf.keras.layers.Embedding(
            params["target_vocab_size"], params["d_embed_dec"]
        )
        self.decoder_positional_encoding = pe.positional_encoding(
            params["max_decode_length"], params["d_embed_dec"]
        )
        self.decoder_dropout = tf.keras.layers.Dropout(params["dropout_dec"])

        self.decoder_stack = transformer.TransformerDecoder(
            {
                "d_embed_dec": params["d_embed_dec"],
                "d_ff": params["d_ff_dec"],
                "dropout": params["dropout_dec"],
                "ff_activation": params["ff_activation_dec"],
                "num_heads": params["num_heads_dec"],
                "num_layers_dec": params["num_layers_dec"],
            }
        )

        self.final_projection = tf.keras.layers.Dense(params["target_vocab_size"])
        self.softmax = tf.keras.layers.Softmax()

    def get_config(self):
        return {"params": self.params}

    def call(self, inputs, training):
        """
        Args:
            inputs: dictionary that contains the following (optional) keys:
                input: ragged int tensor with shape (batch_size, len(params['params_sep_local']), None, None), while the None dimensions are ragged dimensions
                positional_encoding: ragged float tensor with shape (batch_size, len(params['params_sep_local']), None, None, d_embed_enc), while the None dimensions are ragged dimensions, custom positional encoding
                (target: int tensor with shape (batch_size, target_length))
            training: bool, whether model is called in training mode or not
        """
        input = inputs["input"]

        custom_positional_encoding = inputs["positional_encoding"]

        (
            encoder_output,
            padding_mask,
            enc_attn_weights_local,
            enc_attn_weights_global,
        ) = self.encode(input, custom_positional_encoding, training)

        if "target" in inputs:
            target = inputs["target"]
            return self.decode(target, encoder_output, padding_mask, training)
        else:
            return self.predict(
                encoder_output,
                enc_attn_weights_local,
                enc_attn_weights_global,
                padding_mask,
                training,
            )

    def encode(self, inputs, positional_encodings, training):
        """
        Args:
            inputs: int tensor with shape (batch_size, len(params['params_sep_local']), None, None), while the None dimensions are ragged dimensions
            positional_encodings: float tensor with shape (batch_size, len(params['params_sep_local']), None, None, d_embed_enc), while the None dimensions are ragged dimensions, custom postional encoding
            training: boolean, specifies whether in training mode or not
        """
        d_embed_enc = self.params["d_embed_enc"]  # size of embedding vector (internal vector)
        params_sep_local = self.params["params_sep_local"]  # parameters for each separated encoder

        def split_ragged(ragged_tensor, pe_flag):
            """
            Args:
                pe_flag: Whether the ragged tensor is a positional encoding tensor, hence has an embedding dimension at the highest axis and a float type
                ragged_tensor: A ragged tensor with shape (batch_size, len(params['params_sep_local']), None, None, *d_embed_emc*), while the None dimensions are ragged dimensions
            Returns:
                tensor_sep: a list that contains int tensors with shape (batch_size, input_dim[0], input_dim[1], *d_embed_enc*), while input_dim is specified in params['params_sep_local'][i]['input_dimensions'], with i the current index in the list to be returned
            """

            tensors = []
            for i in range(len(params_sep_local)):
                input_dim = params_sep_local[i]["input_dimensions"]
                if pe_flag:
                    shape = [input_dim[0], input_dim[1], d_embed_enc]
                    tensor_type = self.params["dtype_float"]
                else:
                    shape = [input_dim[0], input_dim[1]]
                    tensor_type = self.params["dtype_int"]
                tensors.append(
                    tf.map_fn(
                        lambda x: x[i].to_tensor(),
                        ragged_tensor,
                        fn_output_signature=tf.TensorSpec(shape=shape, dtype=tensor_type),
                    )
                )
            return tensors

        inputs = split_ragged(inputs, pe_flag=False)
        custom_positional_encodings = split_ragged(positional_encodings, pe_flag=True)
        # create embeddings ( later fed into local transformers )
        input_embeddings = []
        padding_masks = []
        for i in range(len(inputs)):
            input = inputs[i]
            params_sep_local = self.params["params_sep_local"][i]
            positional_encoding = custom_positional_encodings[i]
            # create padding masks
            padding_mask = tf.cast(
                tf.math.equal(input, self.params["input_pad_id"]), self.params["dtype_float"]
            )
            padding_masks.append(padding_mask[:, tf.newaxis, tf.newaxis, :, :])

            # create embedding
            input_embedding = self.encoder_embeddings[i](input)
            input_embedding *= tf.math.sqrt(tf.cast(d_embed_enc, self.params["dtype_float"]))
            input_embedding += positional_encoding
            input_embeddings.append(self.encoder_dropouts[i](input_embedding, training=training))

        input_shape = tf.shape(input_embeddings[0])
        batch_size = input_shape[0]
        encoder_outputs_local = []
        padding_masks_local = []
        attn_weights_local = []

        for i in range(len(inputs)):
            input_embedding = input_embeddings[i]
            padding_mask = padding_masks[i]

            # the dimension of one sample
            # the first axis separates the parts which are put independently into the local transformer
            input_dimensions = self.params["params_sep_local"][i]["input_dimensions"]

            # reshape to (batch_size * input_dimensions[0], input_dimensions[1], d_embed_enc)
            input_embedding_local = tf.reshape(
                input_embedding,
                [batch_size * input_dimensions[0], input_dimensions[1], d_embed_enc],
            )
            padding_mask_local = tf.reshape(
                padding_mask, [batch_size * input_dimensions[0], 1, 1, input_dimensions[1]]
            )

            # local encoder step in hierarchical transformer
            encoder_output_local, attn_weights = self.encoders_stack_local[i](
                input_embedding_local, padding_mask_local, training
            )
            attn_weights_local.append(attn_weights)

            # option to only forward one embedding vector each line to the global encoder.
            # Forces the Transformer to learn compression of the local encoder outputs
            if self.params["fix_d1_embed"]:
                encoder_output_local = encoder_output_local[:, 0, :]
                padding_mask_local = padding_mask_local[:, :, :, :, 0]
                input_dimensions[1] = 1

            # reshape to (batch_size, input_dimensions[0] * input_dimensions[1], d_embed_enc)
            encoder_outputs_local.append(
                tf.reshape(
                    encoder_output_local,
                    [batch_size, input_dimensions[0] * input_dimensions[1], d_embed_enc],
                )
            )
            padding_masks_local.append(
                tf.reshape(
                    padding_mask_local,
                    [batch_size, 1, 1, input_dimensions[0] * input_dimensions[1]],
                )
            )

        input_embedding_global = tf.concat(encoder_outputs_local, 1)
        padding_mask_global = tf.concat(padding_masks_local, 3)

        # global encoder step of hierarchical transformer
        encoder_output_global, attn_weights_global = self.encoder_stack_global(
            input_embedding_global, padding_mask_global, training
        )

        return encoder_output_global, padding_mask_global, attn_weights_local, attn_weights_global

    def decode(self, target, encoder_output, input_padding_mask, training):
        """
        Args:
            target: int tensor with shape (bath_size, target_length) including start id at first position
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            input_padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: boolean, specifies whether in training mode or not
        """
        target_length = tf.shape(target)[1]
        look_ahead_mask = transformer.create_look_ahead_mask(
            target_length, self.params["dtype_float"]
        )
        target_padding_mask = transformer.create_padding_mask(
            target, self.params["input_pad_id"], self.params["dtype_float"]
        )
        look_ahead_mask = tf.maximum(look_ahead_mask, target_padding_mask)

        # shift targets to the right, insert start_id at first postion, and remove last element
        target = tf.pad(target, [[0, 0], [1, 0]], constant_values=self.params["target_start_id"])[
            :, :-1
        ]
        target_embedding = self.decoder_embedding(
            target
        )  # (batch_size, target_length, d_embedding)
        target_embedding *= tf.math.sqrt(
            tf.cast(self.params["d_embed_dec"], self.params["dtype_float"])
        )

        target_embedding += self.decoder_positional_encoding[:, :target_length, :]
        decoder_embedding = self.decoder_dropout(target_embedding, training=training)
        decoder_output, attn_weights = self.decoder_stack(
            decoder_embedding, encoder_output, look_ahead_mask, input_padding_mask, training
        )
        output = self.final_projection(decoder_output)
        probs = self.softmax(output)
        return probs, attn_weights

    def predict(
        self,
        encoder_output,
        enc_attn_weights_local,
        enc_attn_weights_global,
        input_padding_mask,
        training,
    ):
        """
        Args:
            encoder_output: float tensor with shape (batch_size, input_length, d_embedding)
            enc_attn_weights_local: dictionary, self attention weights of the encoder
            enc_attn_weights_global: dictionary, self attention weights of the encoder
            input_padding_mask: float tensor with shape (batch_size, 1, 1, input_length)
            training: boolean, specifies whether in training mode or not
        """
        batch_size = tf.shape(encoder_output)[0]

        def logits_fn(ids, i, cache):
            """
            Args:
                ids: int tensor with shape (batch_size * beam_size, index + 1)
                i: int, current index
                cache: dictionary storing encoder output, previous decoder attention values
            Returns:
                logits with shape (batch_size * beam_size, vocab_size) and updated cache
            """
            # set input to last generated id
            decoder_input = ids[:, -1:]
            decoder_input = self.decoder_embedding(decoder_input)
            decoder_input *= tf.math.sqrt(
                tf.cast(self.params["d_embed_dec"], self.params["dtype_float"])
            )
            decoder_input += self.decoder_positional_encoding[:, i : i + 1, :]
            # dropout only makes sense if needs to be tested in training mode
            # think about removing dropout
            decoder_input = self.decoder_dropout(decoder_input, training=training)
            look_ahead_mask = transformer.create_look_ahead_mask(
                self.params["max_decode_length"], self.params["dtype_float"]
            )
            self_attention_mask = look_ahead_mask[:, :, i : i + 1, : i + 1]
            decoder_output, attn_weights = self.decoder_stack(
                decoder_input,
                cache["encoder_output"],
                self_attention_mask,
                cache["input_padding_mask"],
                training,
                cache,
            )

            logits = self.final_projection(decoder_output)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        initial_ids = tf.ones([batch_size], dtype=tf.int32) * self.params["target_start_id"]

        num_heads = self.params["num_heads_dec"]
        d_heads = self.params["d_embed_dec"] // num_heads
        # create cache structure for decoder attention
        cache = {
            "layer_%d"
            % layer: {
                "keys": tf.zeros(
                    [batch_size, 0, num_heads, d_heads], dtype=self.params["dtype_float"]
                ),
                "values": tf.zeros(
                    [batch_size, 0, num_heads, d_heads], dtype=self.params["dtype_float"]
                ),
            }
            for layer in range(self.params["num_layers_dec"])
        }
        # add encoder output to cache
        cache["encoder_output"] = encoder_output
        cache["input_padding_mask"] = input_padding_mask

        beam_search = BeamSearch(logits_fn, batch_size, self.params)
        decoded_ids, scores = beam_search.search(initial_ids, cache)

        if self.params["return_attn_weights"]:

            # computer decoder attention weights
            _, dec_attn_weights = self.decode(
                flatten_beam_dim(decoded_ids), encoder_output, input_padding_mask, training
            )

            return {
                "outputs": decoded_ids,
                "scores": scores,
                "enc_attn_weights_local": enc_attn_weights_local,
                "enc_attn_weights_global": enc_attn_weights_global,
                "dec_attn_weights": dec_attn_weights,
            }

        else:

            return {"outputs": decoded_ids, "scores": scores}
