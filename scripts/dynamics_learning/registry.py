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
            adaptive_history_context=getattr(
                args, "adaptive_history_context", False
            ),
            adaptive_history_short_window=getattr(
                args, "adaptive_history_short_window", 10
            ),
            adaptive_history_mid_window=getattr(
                args, "adaptive_history_mid_window", 25
            ),
            tcnlstm_side_history_scale_init=getattr(
                args, "tcnlstm_side_history_scale_init", 0.05
            ),
            tcnlstm_side_history_selector_prior=getattr(
                args, "tcnlstm_side_history_selector_prior", "uniform"
            ),
            history_context_dim=getattr(args, "history_context_dim", 0),
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
            raw_token_geometric_delta=getattr(
                args, "raw_token_geometric_delta", False
            ),
            adaptive_history_context=getattr(
                args, "adaptive_history_context", False
            ),
            adaptive_history_short_window=getattr(
                args, "adaptive_history_short_window", 10
            ),
            adaptive_history_mid_window=getattr(
                args, "adaptive_history_mid_window", 25
            ),
            history_context_dim=getattr(args, "history_context_dim", 0),
            history_context_mode=getattr(args, "history_context_mode", "none"),
        ),
    }

    selected_model = model[args.model_type]
    print(
        "Model parameters: ",
        sum(p.numel() for p in selected_model.parameters()) / 1000000,
        "M",
    )

    return selected_model
