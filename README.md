# Hangman Model: Finetuning a BERT Model for Token Classification using Character-Level Masked Language Modeling (MLM)

The following is a proposal for training a model to play the Hangman game. In the Hangman game, we typically begin with hidden spaces that correspond to the characters of a word. The objective is to guess the word by making sequential guesses of individual characters. In this particular case, the player is allowed only 6 mistakes before losing the game.

## Proposal

The proposal is to use a bidirectional transformer model (specifically BERT) where the input tokens are characters from words (not subwords as the original BERT). This was achieved by adding spaces between the characters of each word. By considering both directions, it's possible to use the prev and next context to detect a hidden character.

I aimed to finetune a pretrained BERT model to leverage its capabilities in language understanding and add a simple classifier on top of it to predict classes from 0 to 26, corresponding to the alphabet in lowercase. This modified network is called `HangmanNet`.

A BERT architecture is trained with two objectives:

1. Masked Language Modeling (MLM)
2. Next Sentence Prediction (NSP)

For this specific task, we are more interested in the MLM objective, which allows us to mask characters in the word sequence. By randomly masking some characters in the training data, we can simulate scenarios of the Hangman game.

I have prepared the data in that masked format in [DataExploration.ipynb](DataExploration.ipynb), and during training, I used the custom data collator `CustomDataCollatorForMLM`. You can find the training of the `HangmanNet` and `CustomDataCollatorForMLM` in [HangmanNet.ipynb](HangmanNet.ipynb).

**Note:** When preparing the data, if you mask a repeated character, you must mask all other occurrences as well.

Additionally, I used previous guesses as part of the classifier input to improve the model's capabilities. Previous guesses provide strong prior information that we can use to guide model learning. Specifically, this information is concatenated to each of the output embeddings, and feed into the classifier.

## Custom Data Collator for Masking Language Modelling

A custom data collator was implemented to address several issues:

1. Adjust training samples on the fly and generate diversity in the dataset.
2. Maintain a manageable dataset size.
3. Generate masking (hidden characters) on the fly, following a similar approach to Masked Language Modeling (MLM) used in BERT.
4. Different to the original MLM, if I mask a character that is repeated in the word sequence. I have to mask all the ocurrances.

To create the data samples (batches) in the desired format, we utilized the same concept as MLM, which was employed in training the original BERT model. This approach was adapted to randomly mask characters, adhering to the same considerations:

- A percentage of the characters in the word is randomly selected to be masked, given `mlm_probability = 0.5`.
- Of those selected characters, 80% are masked using the masked token `[MASK]`, while 20% are left unchanged.

NOTE: Unlike the original MLM, I decided not to choose a random character for 10% of the characters. This decision was made to avoid potential instabilities, as words with fewer tokens (characters) can behave differently than the longer sentences used to train the original BERT.

## Evaluation Metric

The `HangmanNet` returns a prediction for each masked character in the word sequence, but we only need one at a time when playing the hangman game. Therefore, we created a special `accuracy_unique_char` metric to help us choose only one character from the output predictions of the `HangmanNet`.

I tried different methods for choosing the unique character and found that the best approach is a greedy one, where we select the character with the largest logits (probability) value from all the predicted masked characters. This method yielded an `accuracy_unique_char = 0.70`, so I keep that for the final implementation in the main notebook [hangman_api_user.ipynb](../hangman_api_user.ipynb)

Here are the results for the different methods I tried:

- Random: Test Accuracy Unique Char: 0.6274
- Greedy: Test Accuracy Unique Char: 0.7044
- Random_Greedy: Test Accuracy Unique Char: 0.6698
- Greedy_Random: Test Accuracy Unique Char: 0.6847
- Greedy_Random_Prior: Test Accuracy Unique Char: 0.66

## Additional Notebooks

- The notebook [HangmanCharacterNet.ipynb](HangmanCharacterNet.ipynb) explores an alternative approach using another pretrained model called CharacterNet, which is based on BERT but uses characters as input instead of subwords. However, the results were not significantly different from those obtained using the original BERT model.

- The notebook [HangmanPriorNet.ipynb](HangmanPriorNet.ipynb) investigates the use of additional prior information to improve the model's performance. The approach involves defining a customized loss function based on the relative frequency of characters by word length. This information is valuable because guessing longer words is typically easier than guessing shorter ones.

## Conclusion (Future Work)

I realize that my model performs efficiently when the words are longer because it has more context and fewer patterns to unravel between the guesses. However, my model struggles to predict efficiently the hidden characters in small words. This is primarily because there are many possible patterns in small words, which could match different words.

**Frequencies per word length**

To improve performance with small words, my idea was to use the frequencies of characters per word length. I have noticed in [DataExploration.ipynb](DataExploration.ipynb) that characters are differently distributed per length, giving certain priorities to specific characters (mainly vowels).

My idea was to leverage these relative frequencies to improve and guide the learning of the Hangman neural network. Initially, I tried to add this information as part of the classifier input by concatenating this prior information to the embeddings, similar to how previous guesses were concatenated. This approach can be found in [HangmanPriorNet.ipynb](HangmanPriorNet.ipynb). However, the results were not significantly different.

My future idea is to leverage this prior information to create an auxiliary loss (which will work as a kind of regularizer). This way, I will penalize predictions that are far from these prior probabilities. To control the influence of this auxiliary loss, I will use a parameter to scale the value and perform different tests to check performance.

Note that I also applied Laplacian smoothing to the prior probabilities because some probabilities were set to zero, and saved these probabilities as part of the data in [data/total_rel_freq.csv](character-bert/data/total_rel_freq.csv).

**Other Character-Level Bidirectional Transform**

Additionally, I have tried other models as backbones for the `HangmanNet`. Specifically, I tried `google/canine-s`, which is a network that accepts characters as tokens. You can find my implementation in [HangmanCharacterNet.ipynb](character-bert/HangmanCharacterNet.ipynb). This contrasts with BERT, which accepts words. I added spaces to have one embedding per character. My idea was that a model accepting characters as input could be more efficient in recognizing patterns between characters in a word. However, the results were similar to the BERT backbone.
