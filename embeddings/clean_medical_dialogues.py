import pandas as pd
import re

# import csv file as pandas dataframe
df = pd.read_csv("/Users/dustin/Documents/chatgpt_project/medical_dialogues.csv")

# remove duplicate rows
df = df.drop_duplicates()

# convert rows to strings
df['description'] = df['description'].astype(str)
df['question'] = df['question'].astype(str)
df['answer'] = df['answer'].astype(str)

# remove "Q. " from the description column
df['description'] = df['description'].apply(lambda x: x.replace('Q. ', ''))

# append the description and question columns
df['description_question'] = df['description'] + " " + df['question']
df['description_question_answer'] = df['description'] + " " + df['question'] + " " + df['answer']

# drop the description and question columns
df = df.drop(columns={'description', 'question'})

# count the number of tokens in each answer column, row by row
df['tokens'] = df['answer'].apply(lambda x: len(x.split()))

# if prompt_token_count is greater than 250, drop the row
df = df[df['answer'].apply(lambda x: len(x.split())) <= 250]
df = df[df['answer'].apply(lambda x: len(x.split())) >= 25]

# Define the set of words you want to search for
word_set = set(['Pregnancy','Baby','Mother','Birth','Childbirth','Fetus','Maternal','Newborn','Midwife','Obstetrics','Prenatal',
'Delivery','Labor','Gynecologist','Infant','Postpartum','Breastfeeding','Doula','Ultrasound','C-section','Gestation','Placenta',
'Fertility','Miscarriage','Preterm','Surrogate','Amniotic fluid','Birthing','Colostrum','Embryo','Epidural','Morning sickness',
'Natural childbirth','Nursery','Pregnancy test','Stillbirth','Breech position','Cord blood','Delivery room','Due date','Induction',
'Maternity','Neonate','Perinatal','Postnatal','Pregnancy brain','Pregnancy cravings','Pregnancy fatigue','Pregnancy hormones',
'Pregnancy nausea','Pregnancy sex','Premature birth','Prenatal care','Prenatal vitamins','Sonogram','Trimester','Umbilical cord','VBAC',
'Weight gain','Baby shower','Braxton Hicks contractions','Childbirth education','Cord','Diaper','Ectopic pregnancy','Fetal distress',
'Iron deficiency','Kegel exercises','Lamaze','Leukorrhea','Low birth weight','Mucus plug','Nausea','Obstetrician','Pain relief',
'Placenta previa','Postpartum depression','Pregnancy complications','Pregnancy weight gain','Prenatal massage','Prolactin','Quickening',
'Stretch marks','Stretching','Uterus','Vaginal birth', 'Acupuncture','Amniocentesis','Bed rest','Birth defects','Birth plan','Cervix',
'Contraction','Cravings','Dilation','Doula support','Epidural anesthesia','Family planning','Fetal alcohol syndrome','Folic acid',
'Gestational diabetes','Heartburn','Hemorrhoids','Hyperemesis gravidarum','In vitro fertilization','Infertility','Jaundice','Kick counts',
'Lactation','Miscarriage symptoms','Morning sickness remedies','Nuchal translucency screening','Nurse midwife','Oxytocin','Pregnancy acne',
'Pregnancy anemia','Pregnancy constipation','Pregnancy gingivitis','Pregnancy heartburn','Pregnancy high blood pressure',
'Pregnancy insomnia','Pregnancy leg cramps','Pregnancy mood swings','Pregnancy nosebleeds','Pregnancy nutrition','Pregnancy skin care',
'Pregnancy sleep','Pregnancy snoring','Pregnancy swelling','Pregnancy weight','Pregnancy-induced hypertension','Progesterone',
'Reproductive system','Rh incompatibility','Round ligament pain','Toxemia','Toxoplasmosis','Ultrasound scan','Unplanned pregnancy',
'Urinary incontinence','Uterine contractions','Varicose veins','Zika virus','Alpha-fetoprotein test','Amniotic sac','Anemia','Babies',
'Baby bump','Baby clothes','Baby development','Baby growth','Baby health','Baby movement','Baby names','Baby nutrition','Baby position',
'Baby products','Baby safety','Baby size','Baby sleep','Baby teeth','Belly','Beta hCG test','Birth announcement','Birth center',
'Birth control','Birth order','Birth weight','Bloody show','Body changes','Bottle feeding','Braxton Hicks','Breast changes',
'Breastfeeding diet','Breastfeeding problems','Breastfeeding tips','Caffeine and pregnancy','Carpal tunnel syndrome','Cesarean delivery',
'Childbirth classes','Childbirth complications','Cholestasis of pregnancy','Chorionic villus sampling','Circumcision','Cleft lip',
'Cleft palate','Coccyx','Colic','Common cold','Cord prolapse','Cord wrapped around baby','Couvade syndrome','Delivery complications',
'Delivery position','Delivery process','Depression during pregnancy','Diabetes in pregnancy','Diastasis recti','Dilation and curettage',
'Discharge','Dizziness during pregnancy','Due date calculator','Ectopic pregnancy symptoms','Endometriosis','Episiotomy',
'Exercise during pregnancy','Fetal alcohol effects','Fetal distress','Fetal movements','Fetal surgery','Fetal ultrasound',
'Fetus development','Fetus position','Fetus size','Fetus weight','Fibroids','Flatulence','Fraternal twins','Gender prediction',
'Genetic counseling','Gestational hypertension','Gestational trophoblastic disease','Group B strep','Hair dye',
'Headaches during pregnancy','Health insurance','Healthy pregnancy','Hemorrhage','Hepatitis B','Herpes simplex virus',
'High-risk pregnancy','HIV and pregnancy','Home birth','Hormones in pregnancy','Hot tubs and saunas','Hysterectomy','Hysteroscopy',
'Incompetent cervix','Inducing labor','Infertility treatment','Intrauterine growth restriction','Intrauterine insemination','Iodine',
'Iron-rich foods','Itching during pregnancy','Kegel exercises during pregnancy','Kidney stones and pregnancy','Labor and delivery',
'Labor and delivery process','Lactation consultant','Late-term pregnancy','Lightning crotch','Linea nigra','Lupus and pregnancy',
'Macrosomia','Maternal age','Maternity leave','Molar pregnancy','Monoamniotic twins','Monochorionic twins','Morning sickness causes',
'Nausea and vomiting of pregnancy','Neural tube defect','Nuchal cord','Nursing pads','Obesity and pregnancy','Occiput anterior position',
'Oligohydramnios','Ovarian cysts and pregnancy','gestational', 'preeclampsia', 'prenatally', 'trimesters', 'births', 'colostrum', 
'morning sickness', 'endometriosis', 'physicians', 'latching', 'carrying', 'eclampsia', 'reproductive', 'pregnancy hormones', 'crown', 
'maternity clothes', 'pelvic floor', 'nauseous', 'placenta previa', 'healthy pregnancy', 'placental', 'engorgement', 'fertile', 'doppler', 
'varicose veins', 'baby bump', 'infertility', 'miscarriages', 'uterine', 'antibodies', 'placenta accreta', 'uterus', 'gynecological', 
'multiples', 'delivery room', 'endometrial', 'amniocentesis', 'ectopic pregnancy', 'gestation', 'uterine contractions', 'respiratory', 
'cesarean delivery', 'ob', 'cervical', 'neonatologist', 'vaginal', 'midwifery', 'breastfeeding support', 'stillbirth', 'incontinence', 
'lactation consultant', 'perinatal', 'midwife', 'preconception', 'polyhydramnios', 'oxytocin', 'conceive', 'ovaries', 'second trimester', 
'amniotic', 'preemie', 'folate', 'obstetrical', 'hypertension', 'gynecologists', 'uterus contractions', 'midwives', 'nursing bras', 
'baby girl', 'perinatologist', 'prenatal care', 'endometrium', 'lochia', 'sonographer', 'preconception care', 'embryonic', 'preeclamptic', 
'prenatal exercise', 'umbilical', 'contraception', 'gestational diabetes', 'multiple pregnancies', 'pediatrician', 'premies', 'baby boy', 
'cord blood', 'intrauterine', 'uterine rupture', 'babycenter', 'cesarean section', 'obgyn', 'braxton hicks', 'vaginal birth', 
'fertility treatments', 'oxytocin receptor', 'antenatal', 'placenta accrete', 'prolactin', 'preterm labor', 'miscarriage', 
'fetal distress', 'hormonal', 'placental abruption', 'twin pregnancies', 'childbirth preparation', 'fetal position', 'childbearing', 
'fetal development', 'high risk pregnancy', 'multiparous','folate', 'baby registry', 'dilation', 'in vitro fertilization', 
'inducing labor', 'overdue', 'placental abruption', 'placental insufficiency', 'preconception', 'premature labor', 
'pregnancy announcement', 'pregnancy insomnia', 'pregnancy loss', 'pregnancy pillow', 'pregnancy rhinitis', 'pregnancy snacks', 
'pregnancy symptoms week by week', 'premature rupture of membranes', 'prenatal screening', 'prenatal testing', 'progestin', 
'progesterone suppositories', 'second trimester', 'uterine contraction', 'vaginal discharge', 'baby development', 'baby movement', 
'baby position', 'belly size', 'beta hcg', 'birthing center', 'birthing classes', 'body pillow', 'breast milk storage', 'breast pain', 
'breast pump', 'breastfeeding benefits', 'breastfeeding positions', 'car seat safety', 'childbirth classes', 'contraceptive implant', 
'contraction timer', 'cord prolapse', 'dilation and curettage', 'ectopic pregnancy symptoms', 'episiotomy', 'fetal alcohol syndrome', 
'fetal distress', 'fetal heart rate monitoring', 'fetal movement', 'gestational age', 'gestational diabetes diet', 
'gestational hypertension', 'group b strep', 'home birth', 'hormonal changes during pregnancy', 'hospital bag', 'hospital birth', 
'hospital tour', 'hypnobirthing', 'intrauterine growth restriction', 'lamaze breathing', 'maternity clothes', 'molar pregnancy', 
'morning sickness remedies', 'mucus discharge', 'neonatal abstinence syndrome', 'neonatal intensive care unit', 'newborn feeding', 
'newborn hiccups', 'newborn photography', 'newborn sleep', 'nonstress test', 'nuchal cord', 'overactive bladder', 
'ovarian cysts during pregnancy', 'pelvic pain', 'perineal massage', 'placenta accreta', 'placenta percreta', 
'placental abruption symptoms', 'postpartum belly', 'postpartum bleeding', 'postpartum checkup', 'postpartum constipation', 
'postpartum cramps', 'postpartum depression symptoms', 'postpartum recovery', 'pre eclampsia', 'premature baby development', 
'premature baby milestones', 'premature infant', 'premature labor signs', 'pregnancy and diabetes', 'pregnancy and heartburn', 
'pregnancy and high blood pressure', 'pregnancy and sex', 'pregnancy and teeth', 'pregnancy back pain', 'pregnancy books',
'pregnancy brain fog'])

# Define a function to check if a row contains at least two of the words in the word set
def has_two_words(row):
    count = 0
    for word in word_set:
        if word in row['description_question_answer']:
            count += 1
        if count >= 2:
            return True
    return False

# Filter the DataFrame to only include rows that contain at least two words from the word set
print(len(df))
df = df[df.apply(has_two_words, axis=1)]
print(len(df))

# drop the description_question_answer column
df = df.drop(columns={'description_question_answer'})

# remove "-->" and all the text that follows it in the completion column
df['answer'] = df['answer'].apply(lambda x: x.split('-->')[0])

# remove "(attachment removed to protect patient identity)" in the answer column
df['answer'] = df['answer'].apply(lambda x: x.replace('(attachment removed to protect patient identity)', ''))

# remove "Hi." or "Hello." or "Hello, Welcome to iclinq.com." or "Hello, welcome to icliniq." or "Hello, Welcome back to icliniq.com." in the the answer column
df['answer'] = df['answer'].apply(lambda x: x.replace('Hi. ', ''))

# remove "Hello." in the the answer column
df['answer'] = df['answer'].apply(lambda x: x.replace('Hello. ', ''))

# remove "Hello, Welcome to iclinq.com." in the the answer column
df['answer'] = df['answer'].apply(lambda x: x.replace('Hello, Welcome to iclinq.com. ', ''))

# remove "Hello, welcome to icliniq." in the the answer column
df['answer'] = df['answer'].apply(lambda x: x.replace('Hello, welcome to icliniq. ', ''))

# remove "Hello, Welcome back to icliniq.com." in the the answer column
df['answer'] = df['answer'].apply(lambda x: x.replace('Hello, Welcome back to iclinq.com. ', ''))

# remove "Hello, " in the the answer column
df['answer'] = df['answer'].apply(lambda x: x.replace('Hello, ', ''))

# remove "Hi, " in the the answer column
df['answer'] = df['answer'].apply(lambda x: x.replace('Hi, ', ''))

# remove "I *." in the the answer column
df['answer'] = re.sub("^.*\b(I)\b.*$", "", df['answer'])

# remove unicode characters from the prompt and completion columns
df['description_question'] = df['description_question'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
df['answer'] = df['answer'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

# save dataframe as csv
df.to_csv('/Users/dustin/Documents/chatgpt_project/medical_dialogues_cleaned.csv', index=False)
