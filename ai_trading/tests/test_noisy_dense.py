import unittest
import tensorflow as tf
import numpy as np
from ai_trading.rl.agents.layers.noisy_dense import NoisyDense

class TestNoisyDense(unittest.TestCase):
    """Tests pour la couche NoisyDense"""

    def setUp(self):
        """Configuration initiale pour les tests"""
        tf.random.set_seed(42)
        np.random.seed(42)
        self.batch_size = 8
        self.input_dim = 4
        self.output_dim = 2
        self.inputs = tf.random.normal((self.batch_size, self.input_dim))
        self.sigma_init = 0.5

    def test_initialization(self):
        """Test de l'initialisation de la couche NoisyDense"""
        layer = NoisyDense(self.output_dim, sigma_init=self.sigma_init)
        self.assertEqual(layer.units, self.output_dim)
        self.assertEqual(layer.sigma_init, self.sigma_init)
        
        # Vérifier que la couche est construite correctement
        layer.build((None, self.input_dim))
        self.assertEqual(layer.weight_mu.shape, (self.input_dim, self.output_dim))
        self.assertEqual(layer.weight_sigma.shape, (self.input_dim, self.output_dim))
        self.assertEqual(layer.bias_mu.shape, (self.output_dim,))
        self.assertEqual(layer.bias_sigma.shape, (self.output_dim,))

    def test_output_shape(self):
        """Test de la forme de sortie de la couche NoisyDense"""
        layer = NoisyDense(self.output_dim)
        output = layer(self.inputs)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_training_mode(self):
        """Test que la couche se comporte différemment en mode entraînement et inférence"""
        layer = NoisyDense(self.output_dim)
        
        # Deux passages avec training=True devraient donner des résultats différents
        output1 = layer(self.inputs, training=True)
        output2 = layer(self.inputs, training=True)
        
        # Les sorties doivent être différentes car le bruit est ajouté
        self.assertFalse(tf.reduce_all(tf.equal(output1, output2)))
        
        # En mode inférence, les sorties devraient être identiques
        output3 = layer(self.inputs, training=False)
        output4 = layer(self.inputs, training=False)
        
        self.assertTrue(tf.reduce_all(tf.equal(output3, output4)))

    def test_serialization(self):
        """Test que la couche peut être sérialisée et désérialisée"""
        layer = NoisyDense(self.output_dim, activation='relu', sigma_init=0.1)
        config = layer.get_config()
        
        # Vérifier que les paramètres sont correctement sérialisés
        self.assertEqual(config['units'], self.output_dim)
        self.assertEqual(config['sigma_init'], 0.1)
        
        # Reconstruire la couche à partir de la configuration
        layer2 = NoisyDense.from_config(config)
        self.assertEqual(layer2.units, self.output_dim)
        self.assertEqual(layer2.sigma_init, 0.1)

    def test_model_integration(self):
        """Test que la couche peut être intégrée dans un modèle Keras"""
        # Création d'un modèle simple avec une couche NoisyDense
        inputs = tf.keras.Input(shape=(self.input_dim,))
        outputs = NoisyDense(self.output_dim)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Vérifier que le modèle peut effectuer des prédictions
        predictions = model.predict(self.inputs.numpy())
        self.assertEqual(predictions.shape, (self.batch_size, self.output_dim))

if __name__ == '__main__':
    unittest.main() 