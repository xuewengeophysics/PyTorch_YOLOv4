import models.models as model
if __name__ == '__main__':
    for i, m in enumerate(model.module_list):
        print(m.__class__.__name__)