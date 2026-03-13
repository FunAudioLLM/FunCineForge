from funcineforge.register import tables

@tables.register("face_classes", "FaceRecIR101")
def FaceRecIR101(init_param_path, **kwargs):
    """
    Face embeddings extraction with CurricularFace pretrained model. 
    Reference:
    - https://modelscope.cn/models/iic/cv_ir101_facerecognition_cfglint
    """
    import onnxruntime
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 8
    ort_session = onnxruntime.InferenceSession(
        init_param_path, 
        sess_options=options, 
        providers=['CPUExecutionProvider']
    )
    return ort_session
