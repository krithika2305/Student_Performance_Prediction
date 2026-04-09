import pandas as pd
from pathlib import Path

path = Path('data/student_data.xls')
print('exists', path.exists())
df = None
try:
    df = pd.read_csv(path)
    print('read as csv')
except Exception as e:
    print('csv failed', e)
    import pkgutil
    engine = 'openpyxl' if pkgutil.find_loader('openpyxl') else None
    if engine:
        df = pd.read_excel(path, engine=engine)
        print('read as excel with openpyxl')
    else:
        df = pd.read_excel(path)
        print('read as excel default engine')

cols = [c.strip() for c in df.columns.tolist()]
for i, c in enumerate(cols):
    print(i, repr(c))
print('\n--- values for question-like columns ---')
for raw in df.columns.tolist():
    stripped = raw.strip()
    if any(keyword in stripped for keyword in ['Semester','attendance','internal marks','assignments','participate','study per day','ask doubts','teachers','sleep','mobile','travel','extracurricular','motivated','academic stress','confident','overall academic performance','CGPA']):
        print('===', repr(stripped), '===')
        print(df[raw].dropna().unique())
