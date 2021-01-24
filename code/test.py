import pandas as pd
import pickle
import uvicorn


from preprocessing import preprocess, feature_selection
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

model_path = './data/finalized_model.sav'
train_filepath = './data/TrainingSet.csv'


class MachineParams(BaseModel):
    SlNo: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    V29: float
    V30: float
    V31: float
    V32: float
    V33: float
    V34: float
    V35: float
    V36: float
    V37: float
    V38: float
    V39: float
    V40: float
    V41: float
    V42: float
    V43: float
    V44: float
    V45: float
    V46: float
    V47: float
    V48: float
    V49: float
    V50: float
    V51: float
    V52: float
    V53: float
    V54: float
    V55: float
    V56: float
    V57: float
    V58: float
    V59: float
    V60: float
    V61: float
    V62: float
    V63: float
    V64: float
    V65: float
    V66: float
    V67: float
    V68: float
    V69: float
    V70: float
    V71: float
    V72: float
    V73: float
    V74: float
    V75: float
    V76: float
    V77: float
    V78: float
    V79: float
    V80: float
    V81: float
    V82: float
    V83: float
    V84: float
    V85: float
    V86: float
    V87: float
    V88: float
    V89: float
    V90: float
    V91: float
    V92: float
    V93: float
    V94: float
    V95: float
    V96: float
    V97: float
    V98: float
    V99: float
    V100: float
    V101: float
    V102: float
    V103: float
    V104: float
    V105: float
    V106: float
    V107: float
    V108: float
    V109: float
    V110: float
    V111: float
    V112: float
    V113: float
    V114: float
    V115: float
    V116: float
    V117: float
    V118: float
    V119: float
    V120: float
    V121: float
    V122: float
    V123: float
    V124: float
    V125: float
    V126: float
    V127: float
    V128: float
    V129: float
    V130: float
    V131: float
    V132: float
    V133: float
    V134: float
    V135: float
    V136: float
    V137: float
    V138: float
    V139: float
    V140: float
    V141: float
    V142: float
    V143: float
    V144: float
    V145: float
    V146: float
    V147: float
    V148: float
    V149: float
    V150: float
    V151: float
    V152: float
    V153: float
    V154: float
    V155: float
    V156: float
    V157: float
    V158: float
    V159: float
    V160: float
    V161: float
    V162: float
    V163: float
    V164: float
    V165: float
    V166: float
    V167: float
    V168: float
    V169: float
    V170: float
    V171: float
    V172: float
    V173: float
    V174: float
    V175: float
    V176: float
    V177: float
    V178: float
    V179: float
    V180: float
    V181: float
    V182: float
    V183: float
    V184: float
    V185: float
    V186: float
    V187: float
    V188: float
    V189: float
    V190: float
    V191: float
    V192: float
    V193: float
    V194: float
    V195: float
    V196: float
    V197: float
    V198: float
    V199: float
    V200: float
    V201: float
    V202: float
    V203: float
    V204: float
    V205: float
    V206: float
    V207: float
    V208: float
    V209: float
    V210: float
    V211: float
    V212: float
    V213: float
    V214: float
    V215: float
    V216: float
    V217: float
    V218: float
    V219: float


class Results(BaseModel):
    SlNo: float
    Machine_State: str


@app.post("/api/v1/schema/get-machine-state/")
def get_machine_state(machine_params: MachineParams):
    y_test_pred = pd.DataFrame()
    X, y = preprocess(train_filepath)
    fs, _, _ = feature_selection(X, y, 150)
    df_test = machine_params.dict()
    df_test = pd.DataFrame([df_test])

    X_test = df_test.set_index('SlNo')
    X_test = fs.transform(X_test)

    model = pickle.load(open(model_path, 'rb'))
    y_test_pred['Machine_State'] = model.predict(X_test)
    frame = [df_test[['SlNo']], y_test_pred]
    result = pd.concat(frame, axis=1)
    result['Machine_State'].replace({0: "Good", 1: "Bad"}, inplace=True)

    return JSONResponse(result.to_json(orient='records'))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
