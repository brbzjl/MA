"""For training on NLI data."""
import glovar
from data import nli
from ext import parameters, histories, pickling, training
from models import inference

# Parse configuration settings from command line
params, arg_config = parameters.parse_arguments()


# Get or create History
history = histories.get(
    glovar.PKL_DIR, params.name, params.override, arg_config)


# Report config to be used
config = history.config
print(config)


# Get vocab dict and embeddings
print('Load vocab dict and embedding matrix...')
vocab_dict = pickling.load(glovar.PKL_DIR, 'vocab_dict.pkl')
embedding_matrix = pickling.load(glovar.PKL_DIR, 'glove_embeddings.pkl')[0]


print('Loading data...')
mnli_train = nli.load_json('snli_1.0', 'snli_1.0_test')
snli_train = nli.load_json('snli_1.0', 'snli_1.0_train')
mnli_dev_matched = nli.load_json('snli_1.0', 'snli_1.0_dev')
train_data = nli.NYUDataSet(
    mnli_train, snli_train, vocab_dict, params.train_subset)
tune_data = nli.NLIDataSet(mnli_dev_matched, vocab_dict, params.tune_subset)
train_loader = nli.get_data_loader(train_data, config.batch_size)
dev_loader = nli.get_data_loader(tune_data, config.batch_size)


print('Loading model...')
model = inference.InferenceModel(params.name, config, embedding_matrix)


print('Loading trainer...')
trainer = training.PyTorchTrainer(
    model, history, train_loader, dev_loader, glovar.CKPT_DIR)


print('Training...')
trainer.train()
