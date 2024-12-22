import unittest
from unittest.mock import patch, MagicMock
from main import get_vector_store

class TestMain(unittest.TestCase):

    @patch('main.FAISS')  # Patch the FAISS class from 'main'
    @patch('main.HuggingFaceInstructEmbeddings')  # Patch the correct embedding class
    def test_get_vector_store(self, MockEmbeddings, MockFAISS):
        # Arrange
        mock_embeddings = MagicMock()
        MockEmbeddings.return_value = mock_embeddings
        mock_vector_store = MagicMock()
        MockFAISS.from_texts.return_value = mock_vector_store

        text_chunks = ["chunk1", "chunk2", "chunk3"]

        # Act
        get_vector_store(text_chunks)

        # Assert
        # Accept multiple calls with the same arguments
        MockEmbeddings.assert_any_call(model_name="hkunlp/instructor-large")  # This checks if it's called with this argument at least once
        MockFAISS.from_texts.assert_called_once_with(text_chunks, embedding=mock_embeddings)
        mock_vector_store.save_local.assert_called_once_with("faiss_index")

if __name__ == '__main__':
    unittest.main()
