window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button + TOC sidebar + scroll spy
window.addEventListener('scroll', function() {
    var scrollButton = document.querySelector('.scroll-to-top');
    var tocSidebar = document.getElementById('toc-sidebar');
    var progressBar = document.getElementById('toc-progress-bar');

    // Scroll-to-top button visibility
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }

    // TOC sidebar visibility (show after scrolling past hero)
    if (tocSidebar) {
        if (window.pageYOffset > 400) {
            tocSidebar.classList.add('visible');
        } else {
            tocSidebar.classList.remove('visible');
        }
    }

    // Reading progress bar
    if (progressBar) {
        var docHeight = document.documentElement.scrollHeight - window.innerHeight;
        var scrollPercent = (window.pageYOffset / docHeight) * 100;
        progressBar.style.width = Math.min(scrollPercent, 100) + '%';
    }

    // Scroll spy: highlight current section in TOC
    var sections = document.querySelectorAll('[id^="section-"], #BibTeX');
    var tocLinks = document.querySelectorAll('.toc-link');
    var currentId = '';

    sections.forEach(function(section) {
        var rect = section.getBoundingClientRect();
        if (rect.top <= window.innerHeight * 0.4) {
            currentId = section.id;
        }
    });

    tocLinks.forEach(function(link) {
        link.classList.remove('active');
        if (link.getAttribute('data-section') === currentId) {
            link.classList.add('active');
        }
    });
});

// Autoplay all videos when they scroll into view, pause when they leave.
// This avoids mobile Safari's limit on simultaneous video playback.
function setupVideoAutoplay() {
    const videos = document.querySelectorAll('video');

    if (videos.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                video.play().catch(() => {});
            } else {
                video.pause();
            }
        });
    }, {
        threshold: 0.25
    });

    videos.forEach(video => {
        observer.observe(video);
    });
}

var INTERP_BASE = "https://homes.cs.washington.edu/~kpar/nerfies/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 8000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Pause carousel autoplay when not visible, resume when scrolled into view
    var carouselObserver = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            var carousel = entry.target.bulmaCarousel;
            if (!carousel) return;
            if (entry.isIntersecting) {
                carousel.play();
            } else {
                carousel.pause();
            }
        });
    }, { threshold: 0.2 });

    for (var i = 0; i < carousels.length; i++) {
        carouselObserver.observe(carousels[i].element);
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    setupVideoAutoplay();

    // TOC smooth scroll click handling
    document.querySelectorAll('.toc-link').forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            var targetId = this.getAttribute('data-section');
            var target = document.getElementById(targetId);
            if (target) {
                var offset = target.getBoundingClientRect().top + window.pageYOffset - 40;
                window.scrollTo({ top: offset, behavior: 'smooth' });
            }
        });
    });
})
