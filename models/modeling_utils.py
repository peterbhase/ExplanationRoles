import torch

def custom_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    '''
    copied this from transformers. we have to change the state_dict_keys to allow for loading huggingfance transformers into submodules of models
    '''
    r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

    The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
    To train the model, you should first set it back in training mode with ``model.train()``

    The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
    It is up to you to train those weights with a downstream fine-tuning task.

    The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

    Parameters:
        pretrained_model_name_or_path: either:
          - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
          - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
          - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
          - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
          - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

        model_args: (`optional`) Sequence of positional arguments:
            All remaning positional arguments will be passed to the underlying model's ``__init__`` method

        config: (`optional`) one of:
            - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
            - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`
            Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

        state_dict: (`optional`) dict:
            an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
            This option can be used if you want to create a model from a pretrained configuration but load your own weights.
            In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

        cache_dir: (`optional`) string:
            Path to a directory in which a downloaded pre-trained model
            configuration should be cached if the standard cache should not be used.

        force_download: (`optional`) boolean, default False:
            Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

        resume_download: (`optional`) boolean, default False:
            Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

        proxies: (`optional`) dict, default None:
            A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
            The proxies are used on each request.

        output_loading_info: (`optional`) boolean:
            Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

        kwargs: (`optional`) Remaining dictionary of keyword arguments:
            Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

            - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
            - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

    Examples::

        # For example purposes. Not runnable.
        model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
        model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
        assert model.config.output_attention == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
        model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

    """
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_tf = kwargs.pop("from_tf", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            *model_args,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            **kwargs,
        )
    else:
        model_kwargs = kwargs

    # Load model
    if pretrained_model_name_or_path is not None:
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                # Load from a TF 1.0 checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
            elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                # Load from a TF 2.0 checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    "Error no file named {} found in directory {} or `from_tf` set to False".format(
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"], pretrained_model_name_or_path
                    )
                )
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            archive_file = pretrained_model_name_or_path
        elif os.path.isfile(pretrained_model_name_or_path + ".index"):
            assert (
                from_tf
            ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                pretrained_model_name_or_path + ".index"
            )
            archive_file = pretrained_model_name_or_path + ".index"
        else:
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path, postfix=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME)
            )

        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                msg = "Couldn't reach server at '{}' to download pretrained weights.".format(archive_file)
            else:
                msg = (
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url to model weight files named one of {} but "
                    "couldn't find any such file at this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(cls.pretrained_model_archive_map.keys()),
                        archive_file,
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME],
                    )
                )
            raise EnvironmentError(msg)

        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
    else:
        resolved_archive_file = None

    # Instantiate model.
    model = cls(config, *model_args, **model_kwargs)

    if state_dict is None and not from_tf:
        try:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
        except Exception:
            raise OSError(
                "Unable to load weights from pytorch checkpoint file. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    if from_tf:
        if resolved_archive_file.endswith(".index"):
            # Load from a TensorFlow 1.X checkpoint - provided by original authors
            model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
        else:
            # Load from our TensorFlow 2.0 checkpoints
            try:
                from transformers import load_tf2_checkpoint_in_pytorch_model

                model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
            except ImportError:
                logger.error(
                    "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                    "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                )
                raise
    else:
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: nn.Module, prefix=""):
            import ipdb; ipdb.set_trace()
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

    model.tie_weights()  # make sure word embedding weights are still tied if needed

    # Set model in evaluation mode to desactivate DropOut modules by default
    model.eval()

    if output_loading_info:
        loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
        return model, loading_info

    return model