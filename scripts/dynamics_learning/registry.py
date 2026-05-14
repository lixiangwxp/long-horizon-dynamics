from .models import MLP, LSTM, GRU, TCN, TCNLSTM, GRUTCN


def get_model(args, input_size, output_size):

    model = {
        "mlp": MLP(
            input_size,
            args.history_length,
            args.decoder_sizes,
            output_size,
            args.dropout,
        ),
        "lstm": LSTM(
            input_size,
            args.encoder_sizes,
            args.num_layers,
            args.history_length,
            args.decoder_sizes,
            output_size,
            args.dropout,
            args.encoder_output,
        ),
        "gru": GRU(
            input_size,
            args.encoder_sizes,
            args.num_layers,
            args.history_length,
            args.decoder_sizes,
            output_size,
            args.dropout,
            args.encoder_output,
        ),
        "tcn": TCN(
            input_size,
            args.encoder_sizes,
            args.history_length,
            args.decoder_sizes,
            output_size,
            args.kernel_size,
            args.dropout,
        ),
        "tcnlstm": TCNLSTM(
            input_size,
            args.encoder_sizes,
            args.history_length,
            args.decoder_sizes,
            output_size,
            args.kernel_size,
            args.dropout,
            args.num_layers,
        ),
        "grutcn": GRUTCN(
            input_size,
            args.encoder_sizes,
            args.history_length,
            args.decoder_sizes,
            output_size,
            args.kernel_size,
            args.dropout,
            args.num_layers,
            sampling_frequency=getattr(args, "sampling_frequency", 100),
            multi_step_delta_vomega=getattr(args, "multi_step_delta_vomega", False),
            multi_step_kinematic_update=getattr(
                args, "multi_step_kinematic_update", False
            ),
        ),
    }

    selected_model = model[args.model_type]
    print(
        "Model parameters: ",
        sum(p.numel() for p in selected_model.parameters()) / 1000000,
        "M",
    )

    return selected_model
