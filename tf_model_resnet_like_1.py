class Net(Model):
    '''
    Архитектура развертка в линию. последовательно работает несколько блоков через skip connection
    смотри отличия add и concatenate
    '''
    def __init__(self):
        super().__init__()
        self.layer_dict = {}
        self.layer_dict['embed'] = Embedding(embedding_matrix.shape[0], 
                        word2vec_params['vector_size'],
                        weights=[embedding_matrix],
                        input_length=params['input_length'],
                        trainable=True)
        for block in range(params['blocks']):
            self.layer_dict['conv' + str(block)] = Conv1D(params['kernels'], 12, padding = 'same', activation="selu")
            self.layer_dict['dropout' + str(block)] = Dropout(params['dropout'])
            self.layer_dict['batchnorm' + str(block)] = BatchNormalization()
            if (block > 0) & (block < params['blocks'] - 1):
                self.layer_dict['add' + str(block)] = Add()
        self.layer_dict['flatten'] = Flatten()
        self.layer_dict['dense'] = Dense(params['dense0'], activation=params['activation_dense'])        
        self.layer_dict['dense_output'] = Dense(1, activation=params['activation_dense'])

    def call(self, inputs):
        concat_list = []
        conn = {}
        conn['input'] = inputs
        conn['x'] = self.layer_dict['embed'](conn['input'])
        for block in range(params['blocks']):
            if block == 0:
                conn['conv0'] = self.layer_dict['conv0'](conn['x'])
            elif block == 1:
                conn['conv1'] = self.layer_dict['conv1'](conn['batch0'])
            else:
                conn['conv' + str(block)] = self.layer_dict['conv' + str(block)](conn['add' + str(block - 1)])
            conn['drop' + str(block)] = self.layer_dict['dropout' + str(block)](conn['conv' + str(block)])
            conn['batch' + str(block)] = self.layer_dict['batchnorm' + str(block)](conn['drop' + str(block)])
            if (block > 0) & (block < params['blocks'] - 1):
                conn['add' + str(block)] = self.layer_dict['add' + str(block)]([conn['batch' + str(block)], conn['batch0']])

        conn['flat'] = self.layer_dict['flatten'](conn['batch' + str(params['blocks'] - 1)])
        conn['dense0'] = self.layer_dict['dense'](conn['flat'])
        output = self.layer_dict['dense_output'](conn['dense0'])
        return output
