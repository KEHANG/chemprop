import torch.nn as nn
from torch.autograd import Function

from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.nn_utils import get_activation_function, initialize_weights

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e. outputting
                           learned features in the final layer before prediction.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: TrainArgs):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def featurize(self, *input):
        """
        Computes feature vectors of the input by leaving out the last layer.
        :param input: Input.
        :return: The feature vectors computed by the MoleculeModel.
        """
        return self.ffn[:-1](self.encoder(*input))

    def forward(self, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """
        if self.featurizer:
            return self.featurize(*input)

        output = self.ffn(self.encoder(*input))

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output


class MoleculeModelDANN(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        Initializes the MoleculeModel.

        :param args: Arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e. outputting
                           learned features in the final layer before prediction.
        """
        super(MoleculeModelDANN, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_ffn(args)
        self.create_domain_classifier(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: TrainArgs):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def create_domain_classifier(self, args: TrainArgs):
        """
        Creates the feed-forward network for the domain classifier.

        :param args: Arguments.
        """
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            first_linear_dim = args.hidden_size
            if args.use_input_features:
                first_linear_dim += args.features_size

        activation = get_activation_function(args.activation)

        # Create FFN layers
        ffn = [
            nn.Linear(first_linear_dim, args.ffn_hidden_size),
            activation,
            nn.Linear(args.ffn_hidden_size, self.output_size),
        ]

        # Create FFN model
        self.domain_classifier = nn.Sequential(*ffn)

    def featurize(self, batch, features_batch):
        """
        Computes feature vectors of the input by leaving out the last layer.
        :param input: Input.
        :return: The feature vectors computed by the MoleculeModel.
        """
        return self.ffn[:-1](self.encoder(batch, features_batch))

    def forward(self, batch, features_batch, alpha=0.5):
        """
        Runs the MoleculeModel on input.

        :param input: Molecular input.
        :return: The output of the MoleculeModel. Either property predictions
                 or molecule features if self.featurizer is True.
        """
        if self.featurizer:
            return self.featurize(batch, features_batch)

        feature = self.encoder(batch, features_batch)
        target_output = self.ffn(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            target_output = self.sigmoid(target_output)
            domain_output = self.sigmoid(domain_output)
        if self.multiclass:
            target_output = target_output.reshape((target_output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                target_output = self.multiclass_softmax(target_output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return target_output, domain_output
