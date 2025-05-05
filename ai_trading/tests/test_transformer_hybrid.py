import os
import sys
import tempfile
import unittest

import numpy as np
import pytest
import tensorflow as tf

# Ajouter le répertoire parent au chemin pour importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_hybrid import (
    PositionalEncoding,
    TransformerBlock,
    TransformerGRUModel,
    TransformerLSTMModel,
    create_transformer_hybrid_model,
)

# Activer la mémoire GPU dynamique pour les tests
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Erreur lors de la configuration de la mémoire GPU: {e}")


class TestTransformerHybrid(unittest.TestCase):
    """Tests pour les modèles hybrides Transformer."""

    def setUp(self):
        """Configuration pour les tests."""
        self.batch_size = 8
        self.seq_length = 20
        self.features = 5
        self.embed_dim = 32
        self.num_heads = 4
        self.output_dim = 3

        # Créer une entrée de test
        self.test_input = tf.random.normal(
            (self.batch_size, self.seq_length, self.features)
        )

        # Répertoire temporaire pour enregistrer/charger des modèles
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Nettoyage après les tests."""
        self.temp_dir.cleanup()

    def test_create_transformer_hybrid_model(self):
        """Teste la fonction factory des modèles."""
        # Tester la création d'un modèle GRU
        gru_model = create_transformer_hybrid_model(
            model_type="gru",
            input_shape=(self.seq_length, self.features),
            output_dim=self.output_dim,
        )
        self.assertIsInstance(gru_model, TransformerGRUModel)

        # Tester la création d'un modèle LSTM
        lstm_model = create_transformer_hybrid_model(
            model_type="lstm",
            input_shape=(self.seq_length, self.features),
            output_dim=self.output_dim,
        )
        self.assertIsInstance(lstm_model, TransformerLSTMModel)

        # Tester avec un type non valide
        with self.assertRaises(ValueError):
            create_transformer_hybrid_model(
                model_type="invalid",
                input_shape=(self.seq_length, self.features),
                output_dim=self.output_dim,
            )

    def test_positional_encoding(self):
        """Teste l'encodage positionnel."""
        # Créer un encodage positionnel
        pos_encoding = PositionalEncoding(
            position=self.seq_length, d_model=self.embed_dim
        )

        # Créer une entrée de test avec type float32 pour éviter les problèmes de compatibilité
        test_input = tf.cast(
            tf.random.normal((self.batch_size, self.seq_length, self.embed_dim)),
            dtype=tf.float32,
        )

        # Appliquer l'encodage positionnel
        output = pos_encoding(test_input)

        # Vérifier la forme de sortie
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_length, self.embed_dim)
        )

        # Vérifier que l'encodage positionnel a été ajouté (en convertissant en même type pour la comparaison)
        # Utiliser numpy pour éviter les problèmes de compatibilité de types TensorFlow
        self.assertTrue(np.any(output.numpy() != test_input.numpy()))

    def test_transformer_block(self):
        """Teste le bloc Transformer."""
        # Créer un bloc Transformer
        transformer_block = TransformerBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.embed_dim * 4,
            rate=0.1,
        )

        # Créer une entrée de test
        test_input = tf.random.normal(
            (self.batch_size, self.seq_length, self.embed_dim)
        )

        # Mode entraînement
        train_output = transformer_block(test_input, training=True)
        self.assertEqual(
            train_output.shape, (self.batch_size, self.seq_length, self.embed_dim)
        )

        # Mode inférence
        eval_output = transformer_block(test_input, training=False)
        self.assertEqual(
            eval_output.shape, (self.batch_size, self.seq_length, self.embed_dim)
        )

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
            sequence_length=self.seq_length,
        )

        # Appliquer le modèle à l'entrée de test
        output = model(self.test_input, training=False)

        # Vérifier la forme de sortie
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

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
            sequence_length=self.seq_length,
        )

        # Appliquer le modèle à l'entrée de test
        output = model(self.test_input, training=False)

        # Vérifier la forme de sortie
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_save_load_model(self):
        """Teste la sauvegarde et le chargement du modèle."""
        # Créer un modèle
        model = create_transformer_hybrid_model(
            model_type="gru",
            input_shape=(self.seq_length, self.features),
            output_dim=self.output_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

        # Calculer une sortie initiale
        initial_output = model(self.test_input, training=False).numpy()

        # Définir le chemin de sauvegarde
        save_path = os.path.join(self.temp_dir.name, "test_model.keras")

        # Sauvegarder le modèle
        model.save(save_path)

        # Charger le modèle
        loaded_model = tf.keras.models.load_model(save_path)

        # Calculer la sortie avec le modèle chargé
        loaded_output = loaded_model(self.test_input, training=False).numpy()

        # Vérifier que les sorties sont identiques
        np.testing.assert_allclose(initial_output, loaded_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
