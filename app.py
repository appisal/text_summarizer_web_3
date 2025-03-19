# Function to create share buttons with icons
def create_share_buttons(summary):
    share_links = generate_share_links(summary)

    st.markdown("""
    <style>
        .share-btn-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }
        .share-btn img {
            width: 35px;
            height: 35px;
            transition: transform 0.3s ease-in-out;
        }
        .share-btn img:hover {
            transform: scale(1.2);
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("WhatsApp 📲", key="wa_share"):
            st.markdown(f'<script>window.open("{share_links["WhatsApp"]}");</script>', unsafe_allow_html=True)
    with col2:
        if st.button("Twitter 🐦", key="tw_share"):
            st.markdown(f'<script>window.open("{share_links["Twitter"]}");</script>', unsafe_allow_html=True)
    with col3:
        if st.button("Email ✉️", key="email_share"):
            st.markdown(f'<script>window.open("{share_links["Email"]}");</script>', unsafe_allow_html=True)
    with col4:
        if st.button("LinkedIn 🔗", key="linkedin_share"):
            st.markdown(f'<script>window.open("{share_links["LinkedIn"]}");</script>', unsafe_allow_html=True)
