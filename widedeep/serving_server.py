

import sys
sys.path.append('.')

from serving import MatrixSlowServer


#print(MatrixSlowServer)

serving = MatrixSlowServer(
    host='127.0.0.1:5000', root_dir='.', model_file_name='model.json', weights_file_name='weights.npz')

print('serve start')
serving.serve()
