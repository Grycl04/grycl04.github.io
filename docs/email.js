import emailjs from 'https://cdn.jsdelivr.net/npm/@emailjs/browser@4/dist/email.min.js';

// Initialize EmailJS with your Public Key
emailjs.init({
  publicKey: "bVvu68ObSuhdkBNzj", // from EmailJS dashboard
});

// Optional: send email when KYC is approved or form is submitted
window.sendWelcomeEmail = function(name, email) {
  emailjs.send("service_flz1q7c", "template_5a7ts3q", {
    name: name,
    email: email
  })
  .then(() => {
    alert("✅ Email sent successfully to " + email);
  })
  .catch((error) => {
    console.error("❌ Failed to send email:", error);
    alert("❌ Failed to send email. Please try again later.");
  });
};
