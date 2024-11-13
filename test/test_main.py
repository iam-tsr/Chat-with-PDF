import unittest
from unittest.mock import patch, MagicMock
from main import get_vector_store

# test_main.py


class TestMain(unittest.TestCase):

    @patch('main.FAISS')
    @patch('main.GoogleGenerativeAIEmbeddings')
    def test_get_vector_store(self, MockEmbeddings, MockFAISS):
        # Arrange
        mock_embeddings = MockEmbeddings.return_value
        mock_vector_store = MockFAISS.from_texts.return_value

        text_chunks = ["chunk1", "chunk2", "chunk3"]

        # Act
        get_vector_store(text_chunks)

        # Assert
        MockEmbeddings.assert_called_once_with(model="models/embedding-001")
        MockFAISS.from_texts.assert_called_once_with(text_chunks, embedding=mock_embeddings)
        mock_vector_store.save_local.assert_called_once_with("faiss_index")

if __name__ == '__main__':
    unittest.main()