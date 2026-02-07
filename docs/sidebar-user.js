/**
 * Shared sidebar user profile update - used by buyer and broker pages.
 * Call after sidebar HTML is injected, and when user data is loaded from Firestore.
 */
window.updateSidebarUserProfile = function(userData) {
  if (!userData) {
    userData = (function() {
      try {
        return JSON.parse(localStorage.getItem('userData') || 'null') ||
               JSON.parse(sessionStorage.getItem('userData') || 'null');
      } catch (e) { return null; }
    })();
  }
  if (!userData) return;

  var userNameEl = document.getElementById('userName');
  var userRoleEl = document.getElementById('userRole');
  var userAvatarEl = document.getElementById('userAvatar');

  if (userNameEl) {
    userNameEl.textContent = userData.fullName || userData.displayName || userData.email || 'User';
  }
  if (userRoleEl) {
    var roleMap = {
      landlord: 'Property Owner',
      broker: 'Licensed Broker',
      seller: 'Property Seller',
      admin: 'Administrator',
      agent: 'Property Agent',
      user: 'Buyer'
    };
    userRoleEl.textContent = roleMap[userData.role] || (userData.role === 'agent' ? 'Property Agent' : 'User');
  }
  if (userAvatarEl) {
    var name = userData.fullName || userData.displayName || userData.email || 'U';
    var initials = (name + '').split(' ').map(function(n) { return n ? n[0] : ''; }).join('').toUpperCase().substring(0, 2) || 'U';
    userAvatarEl.innerHTML = '<span style="font-weight: 600;">' + initials + '</span>';
  }
};