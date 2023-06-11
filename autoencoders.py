import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

class BiModalAutoEncoder(tf.keras.Model):
    """
    This class represents a bimodal autoencoder. It extends the tf.keras.Model class
    and combines the encoded representations of two unimodal autoencoders. The encoded
    representatives are concatenated, passed through a fusion encoder and decoded to
    obtain the fused representation.
    
    Parameters
    ----------
    ae1: UniModalAutoencoder()
        Trained autoencoder for the first modality
    ae2: UniModalAutoencoder()
        Trained autoencoder for the second modality
    hidden_dim: Int
        The size of the fusion (encoder/decoder) layers.
    latent_dim: Int
        The size of the latent layer.
        
    Attributes
    ----------
    output_size : Int
        The sum of the input sizes of each modality
    optimizer : tf.Keras.optimizers
        Optimization function for gradient descent
    encoder1: Encoder()
        An alias for the first unimodal autoencoders encoder layer
    encoder2: Encoder()
        An alias for the second unimodal autoencoders encoder layer
    fusion_encoder: Encoder()
        Encoder layer used to transform the latent vectors to bimodal from unimodal
    fusion_decoder: Decoder()
        Decoder layer used to transform the latent vectors back to their original representations.
    
    Methods
    ----------
    call(inputs)
        Forward pass of BiModalAutoEncoder() on given inputs
    train_step(data)
        Override of original train_step in tf.keras.Model
    get_latent_vectors(inputs)
        Returns the bimodal latent vectors as encoded by trained fusion_encoder
    """
    def __init__(self, ae1,ae2, fusion_dim, latent_dim):
        super(BiModalAutoEncoder, self).__init__()
        self.output_size = ae1.input_dim + ae2.input_dim
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.encoder1 = ae1.encoder
        self.encoder2 = ae2.encoder
        self.fusion_encoder = Encoder(fusion_dim, latent_dim)
        self.fusion_decoder = Decoder(fusion_dim, self.output_size)
        
        
    def call(self, inputs):
        """
        Forward pass of BiModalAutoEncoder on the given inputs.

        Parameters
        ----------
        inputs: tuple of tf.Tensor
            Tuple containing two input tensors for the two modalities.

        Returns
        -------
        fusion_decoded: tf.Tensor
            Reconstructed fused representation of the input modalities.
        """
        mod1_input, mod2_input = inputs
        mod1_encoded = self.encoder1(mod1_input)
        mod2_encoded = self.encoder2(mod2_input)
        
        mm_layer = layers.concatenate([mod1_encoded, mod2_encoded])
        
        fusion_encoded = self.fusion_encoder(mm_layer)
        fusion_decoded = self.fusion_decoder(fusion_encoded)
        
        return fusion_decoded
    
    def train_step(self, data):
        """
        Overrides the original train_step in tf.keras.Model.

        Parameters
        ----------
        data: tuple of tf.Tensor
            Tuple containing two input tensors for the two modalities.

        Returns
        -------
        metrics: dict
            Dictionary of metrics for the current training step.
        """
        mod1_input, mod2_input = data[0]
        y = tf.concat([mod1_input,mod2_input], axis=1)
        with tf.GradientTape() as tape:
            y_pred = self([mod1_input, mod2_input], training=True)
            loss = self.compiled_loss(y, y_pred)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y,y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def get_latent_vectors(self, inputs):
        """
        Returns the bimodal latent vectors as encoded by the trained fusion_encoder.

        Parameters
        ----------
        inputs: tuple of tf.Tensor
            Tuple containing two input tensors for the two modalities.

        Returns
        -------
        latent_vectors: np.ndarray
            Array of bimodal latent vectors.
        """
        mod1_input, mod2_input = inputs
        
        mod1_input = tf.expand_dims(mod1_input, axis=0)
        mod2_input = tf.expand_dims(mod2_input, axis=0)
        
          # Add batch dimension
        latent_vectors = []
        for mod1, mod2 in zip(mod1_input, mod2_input):
            mod1_latent = self.encoder1(mod1)
            mod2_latent = self.encoder2(mod2)
            mm_input = tf.concat([mod1_latent, mod2_latent],axis=1)
            fused_latent = self.fusion_encoder(mm_input)
            latent_vectors.append(fused_latent.numpy())
        return np.array(latent_vectors)[0]


class UniModalAutoEncoder(tf.keras.Model):
    """
    This class represents a unimodal autoencoder. It extends the tf.keras.Model class 
    and encapsulates an encoder and decoder. Given an input, it passes it through the encoder 
    to obtain the latent representation and then reconstructs the input using the decoder.
    
    Parameters
    ----------
    input_dim: Int
        The size of the input data
    hidden_dim: Int
        The size of the hidden (encoder/decoder) layers.
    latent_dim: Int
        The size of the latent layer.
        
    Attributes
    ----------

    input_dim: Int
        The size of the input data
    encoder: Encoder()
        Encoder layer used to transform the data into latent vectors
    decoder: Decoder()
        Decoder layer used to transform the latent vectors back to their original representations.
    
    Methods
    ----------
    call(inputs)
        Forward pass of UniModalAutoEncoder()
    get_latent_vectors(inputs)
        Returns the  latent vectors as encoded by trained encoder
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(UniModalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = Encoder(hidden_dim, latent_dim) # Custom Encoder
        self.decoder = Decoder(hidden_dim, input_dim) # Custom Decoder
        
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    
    # Returns latent vectors from the self.encoder output
    def get_latent_vectors(self, inputs):
        """
        Returns the latent vectors as encoded by the trained encoder.
        
        Parameters
        ----------
        inputs: tf.Tensor
            Input data to be encoded.
        
        Returns
        -------
        latent_vectors: np.ndarray
            Latent vectors obtained from the encoder.
        """

        inputs = tf.expand_dims(inputs, axis=0) 
        
        latent_vectors = []
        for mod in inputs:
            latent = self.encoder(mod)
            latent_vectors.append(latent.numpy())
        return np.array(latent_vectors)[0]

# Define Encoder class
class Encoder(layers.Layer):
    """
    This class represents the encoder component of an autoencoder. It consists of two dense layers:
    a hidden layer and a latent layer. The hidden layer performs a nonlinear transformation on the 
    input data using the ReLU activation function, and the latent layer produces the encoded 
    representation of the input data.
    
    Parameters
    ----------
    hidden_dim: int
        The size of the hidden layer.
    latent_dim: int
        The size of the latent layer.
    regularization: tf.keras.regularizers.Regularizer or None, optional
        A kernel regularization function for the hidden layer, or None if no regularization is desired.
    
    Attributes
    ----------
    hidden_layer: tf.keras.layers.Dense
        Encoder layer for nonlinear transformation of input data.
    latent_layer: tf.keras.layers.Dense
        Latent layer.
    """
    def __init__(self, hidden_dim, latent_dim, regularization=None):
        super(Encoder, self).__init__()
        self.hidden_layer = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=regularization)
        self.latent_layer = layers.Dense(latent_dim, activation=None, kernel_regularizer=None)
        
    def call(self, inputs):
        x = self.hidden_layer(inputs)
        encoded = self.latent_layer(x)
        return encoded

# Define Decoder class
class Decoder(layers.Layer):
    """
    This class represents the decoder component of an autoencoder. It consists of two dense layers:
    a hidden layer and an output layer. The hidden layer performs a nonlinear transformation on the 
    encoded data using the ReLU activation function, and the output layer reconstructs the original 
    input data.
    
    Parameters
    ----------
    hidden_dim: int
        The size of the hidden layer.
    output_dim: int
        The size of the output layer.
    regularization: tf.keras.regularizers.Regularizer or None, optional
        A kernel regularization function for the hidden layer, or None if no regularization is desired.
    
    Attributes
    ----------
    hidden_layer: tf.keras.layers.Dense
        Decoder layer for nonlinear transformation of encoded data.
    output_layer: tf.keras.layers.Dense
        Output layer for reconstructing the original input data.
    """
    def __init__(self, hidden_dim, output_dim, regularization=None):
        super(Decoder, self).__init__()
        self.hidden_layer = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=regularization)
        self.output_layer = layers.Dense(output_dim, activation=None, kernel_regularizer=None)
    
    def call(self, inputs):
        """
        Forward pass of the decoder.
        
        Parameters
        ----------
        inputs: tf.Tensor
            Encoded data to be decoded.
            
        Returns
        -------
        decoded: tf.Tensor
            Reconstructed original input data.
        """
        x = self.hidden_layer(inputs)
        decoded = self.output_layer(x)
        return decoded 




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# class MultiModalAE(tf.keras.Model):
#     def __init__(self, latent_dim, num_genes, num_peaks):
#         super(MultiModalAE, self).__init__()
        
#         self.latent_dim = latent_dim
#         self.num_genes = num_genes
#         self.num_peaks = num_peaks
        
#         # Define the layers
#         self.fusion_encoder = layers.Dense(1280, activation='relu')
#         self.dropout_encoder = layers.Dropout(0.1)
#         self.latent_layer = layers.Dense(latent_dim)
#         self.dropout_decoder = layers.Dropout(0.1)
#         self.fusion_decoder = layers.Dense(1280, activation='relu')
#         self.rna_output = layers.Dense(num_genes)
#         self.atac_output = layers.Dense(num_peaks)
    
#     def call(self, inputs):
#         rna_input, atac_input = inputs  # Unpack the inputs
        
#         # Concatenate the input tensors
#         mm_layer = layers.concatenate([rna_input, atac_input])
        
#         # Build autoencoder
#         fusion_encoder_output = self.fusion_encoder(mm_layer)
#         drop_encoder_output = self.dropout_encoder(fusion_encoder_output)
#         latent_layer_output = self.latent_layer(drop_encoder_output)
#         drop_decoder_output = self.dropout_decoder(latent_layer_output)
#         fusion_decoder_output = self.fusion_decoder(drop_decoder_output)
#         rna_output = self.rna_output(fusion_decoder_output)
#         atac_output = self.atac_output(fusion_decoder_output)
        
#         return rna_output, atac_output

#     def train_step(self, data):
#         rna_input, atac_input = data[0]  # Unpack the data

#         with tf.GradientTape() as tape:
#             # Forward pass
#             rna_output, atac_output = self([rna_input, atac_input], training=True)
#             rna_reconstruction_loss = self.rna_loss(rna_input, rna_output)
#             atac_reconstruction_loss = self.atac_loss(atac_input, atac_output)
#             total_loss = rna_reconstruction_loss + atac_reconstruction_loss

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(total_loss, trainable_vars)
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))

#         # Update metrics
#         self.compiled_metrics.update_state([rna_input, atac_input], [rna_output, atac_output])

#         # Return the metrics as a dictionary
#         return {m.name: m.result() for m in self.metrics}

#     def metrics(self):
#         return [self.loss_tracker, ]

    
    
#     def get_latent_vectors(self, mm_input, as_array=False):
        
#         fusion_encoder = self.fusion_encoder
#         latent_layer = self.latent_layer

#         mm_input = tf.expand_dims(mm_input, axis=0)  # Add batch dimension
#         fusion_encoder_output = fusion_encoder(mm_input)
#         latent_vector = latent_layer(fusion_encoder_output)
#         if as_array:
#             return np.array(latent_vector)
#         return latent_vector

# class LatentClassifier(tf.keras.Model):
#     def __init__(self, hidden_dim, num_classes, autoencoder, dropout_rate=0.5, l2_reg=0.01):
#         super(LatentClassifier, self).__init__()
#         self.autoencoder = autoencoder
#         self.autoencoder._trainable = False
#         self.hidden_dim = hidden_dim
#         self.latent_dim = self.autoencoder.latent_dim
#         self.input_layer = layers.Input(self.latent_dim)
#         self.first_layer = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=l2(l2_reg))
#         self.dropout_layer = layers.Dropout(dropout_rate)
#         self.second_layer = layers.Dense(hidden_dim, activation='relu', kernel_regularizer=l2(l2_reg))
#         self.output_layer = layers.Dense(num_classes+1)
    
#     def call(self, inputs):
#         rna_input, atac_input = inputs
        
#         encoded_tensors = self.get_encoded_tensors(rna_input, atac_input)
#         dense_layer = self.first_layer(encoded_tensors)
#         dense_layer = self.dropout_layer(dense_layer)
#         dense_layer = self.second_layer(dense_layer)
#         output_layer = self.output_layer(dense_layer)
        
#         return output_layer
    
#     def train_step(self, data):
#         x, y = data
        
#         with tf.GradientTape() as tape:
#             predictions = self(x, training=True)
#             loss = self.compiled_loss(y, predictions)
        
#         gradients = tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#         self.compiled_metrics.update_state(y, predictions)
        
#         return {m.name: m.result() for m in self.metrics}

    
#     def get_encoded_tensors(self, rna_sample, atac_sample):
#         inputs = [rna_sample, atac_sample]
#         encoded_tensor = self.autoencoder(inputs)[1]  # Get the latent vectors
#         return encoded_tensor

    
    
