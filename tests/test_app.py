import pytest
from unittest.mock import patch, MagicMock, mock_open

# Import the target module and functions to be tested
import src.app.main as app_main
from src.app.main import main, get_response

def test_main_execution_and_interactive_loop():
    """
    Smoke test to ensure the main function can execute and the interactive loop works.
    Mocks external dependencies and simulates 'quit' to exit the loop.
    """
    # We patch the objects directly on the imported module `app_main` to avoid string resolution issues.
    with patch.object(app_main, 'TrainingService') as mock_TrainingService, \
         patch.object(app_main, 'SentencePieceTokenizer') as mock_SentencePieceTokenizer, \
         patch.object(app_main, 'ChatDataset') as mock_ChatDataset, \
         patch.object(app_main, 'ChatModel') as mock_ChatModel, \
         patch.object(app_main, 'GenerationService') as mock_GenerationService, \
         patch.object(app_main, 'glob') as mock_glob, \
         patch.object(app_main.os.path, 'exists') as mock_exists, \
         patch('builtins.open', mock_open(read_data="User: Hello\nAssistant: Hi")) as mock_file, \
         patch('builtins.input', side_effect=['quit']) as mock_input:

        # --- Setup Mocks ---
        mock_glob.glob.return_value = ['dummy_data.txt']
        mock_exists.return_value = True  # Assume tokenizer exists to simplify the test

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.get_vocab_size.return_value = 512
        mock_SentencePieceTokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.count_parameters.return_value = (1000, 1000)
        mock_ChatModel.return_value = mock_model_instance

        mock_generation_service_instance = MagicMock()
        mock_GenerationService.return_value = mock_generation_service_instance

        mock_trainer_instance = MagicMock()
        mock_TrainingService.return_value = mock_trainer_instance

        # --- Execute ---
        try:
            # We call the imported main function directly
            main()
        except Exception as e:
            pytest.fail(f"main() raised an unexpected exception: {e}")

        # --- Assertions ---
        # Assert that the major components were initialized as expected
        mock_glob.glob.assert_called_once_with("./data/*.txt")
        mock_exists.assert_called_once_with("tokenizer.model")
        mock_SentencePieceTokenizer.assert_called_once()
        mock_ChatDataset.assert_called()
        mock_ChatModel.assert_called_once()
        mock_TrainingService.assert_called_once()
        mock_GenerationService.assert_called()

        # Assert that the training process was initiated
        mock_trainer_instance.train.assert_called_once()

        # Assert that the interactive loop was entered and input was requested
        mock_input.assert_called_once()


def test_get_response_success():
    """Tests the get_response function for a successful generation."""
    mock_generation_service = MagicMock()
    mock_generation_service.generate.return_value = "This is a test response."

    prompt = "User: Test prompt"
    response = get_response(mock_generation_service, prompt)

    assert response == "This is a test response."
    mock_generation_service.generate.assert_called_once_with(
        prompt, max_length=50, temperature=0.7
    )

def test_get_response_error():
    """Tests the get_response function when the generation process fails."""
    mock_generation_service = MagicMock()
    mock_generation_service.generate.side_effect = Exception("Generation failed")

    prompt = "User: Test prompt"
    response = get_response(mock_generation_service, prompt)

    assert response == "Sorry, I encountered an error."