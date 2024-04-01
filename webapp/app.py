from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from werkzeug.utils import secure_filename
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

heart_model = pickle.load(open('webapp/model/heart.pkl', 'rb'))
bp_model = pickle.load(open('webapp/model/bp.pkl', 'rb'))
diabetes_model = pickle.load(open('webapp/model/diabetes.pkl', 'rb'))
bc_model = pickle.load(open('webapp/model/bc.pkl', 'rb'))
kd_model = load_model('webapp/model/kd.h5')
# print(kd_model.summary())
covid_model = load_model('webapp/model/covid.h5')


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/heart_page')
def heart():
    return render_template('heart.html')


@app.route('/diabetes_page')
def diabeties():
    return render_template('diabetes.html')


@app.route('/bc_page')
def bc():
    return render_template('bc.html')


@app.route('/bp_page')
def bp():
    return render_template('bp.html')


@app.route('/kidney_page')
def kidney():
    return render_template('kidney.html')


@app.route('/covid_page')
def covid():
    return render_template('covid.html')


@app.route('/heart_predict', methods=['POST', 'GET'])
def heart_predict():
    age = request.form.get("age")
    sex = request.form.get("sex")
    cp = request.form.get("cp")
    re = request.form.get("re")
    ch = request.form.get("ch")
    fa = request.form.get("fa")
    rc = request.form.get("rc")
    mhr = request.form.get("mhr")
    ex = request.form.get("ex")
    op = request.form.get("op")
    sl = request.form.get("sl")
    ca = request.form.get("ca")
    th = request.form.get("th")

    final_heart = [int(age), int(sex), int(cp), int(re), int(ch), int(
        fa), int(rc), int(mhr), int(ex), float(op), int(sl), int(ca), int(th)]
    result_heart = heart_model.predict([final_heart])

    if result_heart == [1]:
        return render_template('heart.html', heart_prediction_text='High Risk Of Heart Disease')
    else:
        return render_template('heart.html', heart_prediction_text="Low Risk Of Heart Disease")


@app.route('/bp_predict', methods=['POST', 'GET'])
def bp_predict():
    pn = request.form.get("pn")
    loh = request.form.get("loh")
    gpc = request.form.get("gpc")
    ag = request.form.get("ag")
    bmi = request.form.get("bmi")
    se = request.form.get("se")
    pr = request.form.get("pr")
    sm = request.form.get("sm")
    pa = request.form.get("pa")
    scind = request.form.get("scind")
    acpd = request.form.get("acpd")
    los = request.form.get("los")
    ckd = request.form.get("ckd")
    aatd = request.form.get("aatd")
    final_bp = [int(pn), float(loh), float(gpc), int(ag), int(bmi), int(se), int(
        pr), int(sm), int(pa), int(scind), int(acpd), int(los), int(ckd), int(aatd)]
    result_bp = bp_model.predict([final_bp])

    if result_bp == [1]:
        return render_template('bp.html', bp_prediction_text="Blood Pressure is not into the normal range")
    else:
        return render_template('bp.html', bp_prediction_text="Blood Pressure is into the normal range")


@app.route('/diabetes_predict', methods=['POST', 'GET'])
def diabetes_predict():
    pr = request.form.get("pr")
    gc = request.form.get("gc")
    bp = request.form.get("bp")
    st = request.form.get("st")
    ins = request.form.get("ins")
    bmi = request.form.get("bmi")
    dpf = request.form.get("dpf")
    age = request.form.get("age")
    final_diabetes = [int(pr), int(gc), int(bp), int(
        st), int(ins), float(bmi), float(dpf), int(age)]
    result_diabetes = diabetes_model.predict([final_diabetes])

    if result_diabetes == [1]:
        return render_template('diabetes.html', diabetes_prediction_text="High Risk Of Diabetes")
    else:
        return render_template('diabetes.html', diabetes_prediction_text="Low Risk Of Diabetes")


@app.route('/bc_predict', methods=['POST', 'GET'])
def bc_predict():
    mr = request.form.get("mr")
    mt = request.form.get("mt")
    mp = request.form.get("mp")
    ma = request.form.get("ma")
    ms = request.form.get("ms")
    mc = request.form.get("mc")
    mcv = request.form.get("mcv")
    mcp = request.form.get("mcp")
    msy = request.form.get("msy")
    mfd = request.form.get("mfd")
    re = request.form.get("re")
    te = request.form.get("te")
    pe = request.form.get("pe")
    ae = request.form.get("ae")
    se = request.form.get("se")
    ce = request.form.get("ce")
    cve = request.form.get("cve")
    cpe = request.form.get("cpe")
    sye = request.form.get("sye")
    fde = request.form.get("fde")
    wr = request.form.get("wr")
    wt = request.form.get("wt")
    wp = request.form.get("wp")
    wa = request.form.get("wa")
    ws = request.form.get("ws")
    wc = request.form.get("wc")
    wcv = request.form.get("wcv")
    wcp = request.form.get("wcp")
    wsy = request.form.get("wsy")
    wfd = request.form.get("wfd")
    final_bc = [float(mr), float(mt), float(mp), float(ma), float(ms), float(mc), float(mcv), float(mcp), float(msy), float(mfd), float(re), float(te), float(pe), float(ae), float(
        se), float(ce), float(cve), float(cpe), float(sye), float(fde), float(wr), float(wt), float(wp), float(wa), float(ws), float(wc), float(wcv), float(wcp), float(wsy), float(wfd)]
    result_bc = bc_model.predict([final_bc])

    if result_bc == [1]:
        return render_template('bc.html', bc_prediction_text="High Risk Of The Breast Cancer is Benign")
    else:
        return render_template('bc.html', bc_prediction_text="High Risk Of The Breast Cancer is Malignant")


@app.route('/kd_predict', methods=['POST', 'GET'])
def kd_predict():
    sg = request.form.get("sg")
    al = request.form.get("al")
    sc = request.form.get("sc")
    hemo = request.form.get("hemo")
    pcv = request.form.get("pcv")
    htn = request.form.get("htn")
    final_kd = [float(sg), float(al), float(
        sc), float(hemo), int(pcv), int(htn)]
    result_kd = kd_model.predict([final_kd])

    if result_kd >= 0.5:
        return render_template('kidney.html', kd_prediction_text='High Risk Of Kidney Disease')
    else:
        return render_template('kidney.html', kd_prediction_text='Low Risk Of Kidney Disease')


def covid_predict(imgpath, model):
    img = load_img(imgpath, target_size=(100, 100))
    x = img_to_array(img)
    x = np.reshape(x, (-1, 100, 100, 3))
    result = model.predict(x)
    if result[0][0] == 0.0:
        return "Low Risk Of The Patient Suffering with COVID-19"
    else:
        return "High Risk Of The Patient Suffering with COVID-19"


@app.route('/covid_prediction', methods=['POST', 'GET'])
def covid_prediction():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'COVID-19_Radiography_Dataset\\COVID\\images', secure_filename(f.filename))
        # file_path = os.path.join(
        #     basepath, 'COVID-19_Radiography_Dataset\\Normal\\images', secure_filename(f.filename))
        f.save(file_path)

        preds = covid_predict(file_path, covid_model)
        return str(preds)
    return None


if __name__ == '__main__':
    app.run(debug=True)
