// auth-check.js - Save this in your project root
import { auth, db } from './firebase-common.js';
import { doc, getDoc } from "https://www.gstatic.com/firebasejs/12.4.0/firebase-firestore.js";
import { signOut } from "https://www.gstatic.com/firebasejs/12.4.0/firebase-auth.js";

// Function to verify user role and redirect if needed
export async function verifyUserRole(requiredRole) {
  try {
    const user = auth.currentUser;
    
    if (!user) {
      // User not logged in, redirect to login
      window.location.href = '../login.html';
      return null;
    }

    // Get user data from Firestore
    const userDoc = await getDoc(doc(db, "users", user.uid));
    
    if (!userDoc.exists()) {
      console.error('User document not found');
      await signOut(auth);
      window.location.href = '../login.html';
      return null;
    }

    const userData = userDoc.data();
    const userRole = userData.role || 'user';
    
    console.log('User role:', userRole, 'Required role:', requiredRole);
    
    // Check if user has the required role for this dashboard
    if (userRole !== requiredRole) {
      // Redirect to their appropriate dashboard
      const dashboardMap = {
        broker: '../broker/broker_dashboard.html',
        seller: '../seller/seller_dashboard.html',
        landlord: '../landlord/landlord_dashboard.html',
        agent: '../broker/agent_dashboard.html',
        admin: '../admin/dashboard.html'
      };
      
      if (dashboardMap[userRole]) {
        console.log(`Redirecting ${userRole} to ${dashboardMap[userRole]}`);
        window.location.href = dashboardMap[userRole];
      } else {
        // Unknown role, log out
        await signOut(auth);
        window.location.href = '../login.html';
      }
      return null;
    }
    
    return { user, userData };
    
  } catch (error) {
    console.error('Auth check error:', error);
    window.location.href = '../login.html';
    return null;
  }
}

// Function to get current user data
export async function getCurrentUserData() {
  try {
    const user = auth.currentUser;
    if (!user) return null;
    
    const userDoc = await getDoc(doc(db, "users", user.uid));
    return userDoc.exists() ? userDoc.data() : null;
  } catch (error) {
    console.error('Error getting user data:', error);
    return null;
  }
}

// Function to check if user is authenticated (any role)
export function isAuthenticated() {
  try {
    const user = auth.currentUser;
    return user !== null;
  } catch (error) {
    console.error('Auth check error:', error);
    return false;
  }
}

// Function to logout user
export async function logoutUser() {
  try {
    await signOut(auth);
    window.location.href = '../login.html';
  } catch (error) {
    console.error('Logout error:', error);
    window.location.href = '../login.html';
  }
}