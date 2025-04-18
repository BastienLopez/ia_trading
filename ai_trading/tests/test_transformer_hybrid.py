import unittest
import numpy as np
import tensorflow as tf
import os
import sys
import tempfile

# Ajouter le répertoire parent au chemin pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.models.transformer_hybrid import (
    TransformerBlock,
    PositionalEncoding,
    TransformerGRUModel,
    TransformerLSTMModel,
    create_transformer_hybrid_model
)

class TestTransformerHybrid(unittest.TestCase):
    """Tests pour les modèles hybrides Transformer."""
    
    def setUp(self):
        """Initialise les données de test."""
        # Paramètres de test
        self.batch_size = 8
        self.seq_length = 20
        self.features = 10
        self.output_dim = 5
        self.embed_dim = 32
        self.num_heads = 4
        
        # Créer des données de test
        self.test_input = np.random.random((self.batch_size, self.seq_length, self.features))
        
    def test_transformer_block(self):
        """Teste le bloc Transformer."""
        # Créer un bloc Transformer
        transformer_block = TransformerBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            rate=0.1
        )
        
        # Créer une entrée de test (déjà dans l'espace d'embedding)
        test_input = tf.random.normal((self.batch_size, self.seq_length, self.embed_dim))
        
        # Appliquer le bloc Transformer
        output = transformer_block(test_input, training=False)
        
        # Vérifier la forme de sortie
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.embed_dim))
        
    def test_positional_encoding(self):
        """Teste l'encodage positionnel."""
        # Créer un encodage positionnel
        pos_encoding = PositionalEncoding(position=self.seq_length, d_model=self.embed_dim)
        
        # Créer une entrée de test
        test_input = tf.random.normal((self.batch_size, self.seq_length, self.embed_dim))
        
        # Appliquer l'encodage positionnel
        output = pos_encoding(test_input)
        
        # Vérifier la forme de sortie
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.embed_dim))
        
        # Vérifier que l'encodage positionnel a été ajouté
        self.assertTrue(tf.reduce_any(output != test_input))
        
    def test_transformer_gru_model(self):
        """Teste le modèle hybride Transformer-GRU."""
        # Créer un modèle Transformer-GRU
        model = TransformerGRUModel(
            input_shape=(self.seq_length, self.features),
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            num_transformer_blocks=2,
            gru_units=self.embed_dim,
            dropout_rate=0.1,
            recurrent_dropout=0.0,
            sequence_length=self.seq_length
        )
        
        # Appliquer le modèle à l'entrée de test
        output = model(self.test_input, training=False)
        
        # Vérifier la forme de sortie
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Vérifier la capacité d'entraînement
        with tf.GradientTape() as tape:
            output = model(self.test_input, training=True)
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))
        
    def test_transformer_lstm_model(self):
        """Teste le modèle hybride Transformer-LSTM."""
        # Créer un modèle Transformer-LSTM
        model = TransformerLSTMModel(
            input_shape=(self.seq_length, self.features),
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            num_transformer_blocks=2,
            lstm_units=self.embed_dim,
            dropout_rate=0.1,
            recurrent_dropout=0.0,
            sequence_length=self.seq_length
        )
        
        # Appliquer le modèle à l'entrée de test
        output = model(self.test_input, training=False)
        
        # Vérifier la forme de sortie
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Vérifier la capacité d'entraînement
        with tf.GradientTape() as tape:
            output = model(self.test_input, training=True)
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))
        
    def test_create_transformer_hybrid_model(self):
        """Teste la fonction de création de modèle hybride."""
        # Créer un modèle GRU
        gru_model = create_transformer_hybrid_model(
            model_type="gru",
            input_shape=(self.seq_length, self.features),
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            num_transformer_blocks=2,
            rnn_units=self.embed_dim,
            dropout_rate=0.1,
            recurrent_dropout=0.0,
            sequence_length=self.seq_length
        )
        
        # Vérifier le type du modèle
        self.assertIsInstance(gru_model, TransformerGRUModel)
        
        # Créer un modèle LSTM
        lstm_model = create_transformer_hybrid_model(
            model_type="lstm",
            input_shape=(self.seq_length, self.features),
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            num_transformer_blocks=2,
            rnn_units=self.embed_dim,
            dropout_rate=0.1,
            recurrent_dropout=0.0,
            sequence_length=self.seq_length
        )
        
        # Vérifier le type du modèle
        self.assertIsInstance(lstm_model, TransformerLSTMModel)
        
        # Tester avec un type de modèle invalide
        with self.assertRaises(ValueError):
            create_transformer_hybrid_model(
                model_type="invalid",
                input_shape=(self.seq_length, self.features),
                output_dim=self.output_dim
            )
    
    def test_save_load_model(self):
        """Teste la sauvegarde et le chargement du modèle."""
        # Créer un modèle
        model = create_transformer_hybrid_model(
            model_type="gru",
            input_shape=(self.seq_length, self.features),
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )
        
        # Calculer une sortie initiale
        initial_output = model(self.test_input, training=False).numpy()
        
        # Sauvegarder le modèle dans un répertoire temporaire
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "transformer_hybrid_model")
            model.save_weights(model_path)
            
            # Créer un nouveau modèle avec la même configuration
            new_model = create_transformer_hybrid_model(
                model_type="gru",
                input_shape=(self.seq_length, self.features),
                output_dim=self.output_dim,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads
            )
            
            # Charger les poids
            new_model.load_weights(model_path)
            
            # Calculer la sortie avec le nouveau modèle
            new_output = new_model(self.test_input, training=False).numpy()
            
            # Vérifier que les sorties sont identiques
            np.testing.assert_allclose(initial_output, new_output, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    unittest.main() 