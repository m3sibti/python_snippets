
# GPU related information

The following library is helpful for monitoring in GPU utilization, you can call it in custom callbacks `on_batch_begin/end`
https://github.com/anderskm/gputil

**Calculate Keras's Model Memory Usage:**
    
    def get_model_memory_usage(batch_size, model):
        shapes_mem_count = 0
        internal_model_mem_count = 0
        for l in model.layers:
            layer_type = l.__class__.__name__
            if layer_type == 'Model':
                internal_model_mem_count += get_model_memory_usage(batch_size, l)
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem
    
        trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    
        number_size = 4.0
        if K.floatx() == 'float16':
             number_size = 2.0
        if K.floatx() == 'float64':
             number_size = 8.0
    
        total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
        mbytes = np.round(total_memory / (1024.0 ** 2), 3) + internal_model_mem_count
        #  gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
        # return # gbytes
        print(f"Model Types = {number_size}, {K.floatx()}")
        print(f"Batch size = {batch_size}")
        print(f"Shapes Memory = {shapes_mem_count}")
        print(f"Params: \n    - Trainable = {trainable_count}\n    - Non-Trainable {non_trainable_count}")
        print(f"\nTotal Memory Required = {total_memory} bytes, {mbytes}MB")
        print(f"Approx, Actual Memory Required = {mbytes*3}MB, (Memory x 3) 3 for grads and momentum variables")
