import streamlit

num_epochs = 1000
learning_rate = 0.001
batch_size = 64
multiplier = 1.0
patience = 10
p = 0.0001
latent_size = 16

# File uploaders for user to upload their own data files
file_1 = st.file_uploader("Choose the first Excel file (Correct data)", type="xlsx")
file_2 = st.file_uploader("Choose the second Excel file (Wrong data)", type="xlsx")
