;;;--------------------------------------------------------------------------
;;; 
;;; casc customizations for .emacs file. can be used with emacs or xemacs.
;;;
;;; to use this file just by itself, do
;;;     emacs  -l sample.emacs file_to_edit
;;;
;;; to get emacs to automatically use this file on startup, add this to
;;; your .emacs file.
;;;
;;; author: chandrika kamath
;;;         lawrence livermore national laboratory
;;;         livermore, ca 94551
;;;
;;; date: october 24, 1997
;;;
;;; modified:  1) Added the font lock for c++ mode.
;;;               Added indent-tabs-mode to force indentation with spaces
;;;               chandrika kamath, december 9, 1997.
;;;--------------------------------------------------------------------------

;;;
;;; C/C++ mode
;;;
;;; c-auto-newline automatically inserts a newline before and after a
;;; brace, and after a ";". To toggle this on/off, use C-x C-a. To 
;;; turn it off permanently, comment out the line that sets c-auto-newline.
;;; 
;;; the ellemtel style sets the opening brace on a new line. to set the
;;; brace at the end of the same line, use the stroustrup style.
;;;

(defun my-c-mode-common-hook ()
  (c-set-style "ellemtel")
  (setq c-tab-always-indent t)
  (setq c-basic-offset 3)
  (c-set-offset 'knr-argdecl-intro 0)
  (setq c-auto-newline t)
  (setq c-continued-statement-offset 3)
  (setq c-brace-offset 0)
  (c-set-offset 'inclass '+)
  (setq line-number-mode t)
  (setq indent-tabs-mode nil)
)

(add-hook 'c-mode-common-hook 'my-c-mode-common-hook)

;;;
;;; to turn on color in cc-mode (to set background color, see the manpages for
;;; emacs or xemacs)
;;;

(add-hook 'c-mode-hook 'turn-on-font-lock)
(add-hook 'c++-mode-hook 'turn-on-font-lock)
(setq font-lock-maximum-decoration '((c-mode . 3) (c++-mode . 3)))

;;;
;;; fortran mode (f77)
;;;

(setq auto-mode-alist (cons '("\.for$" . fortran-mode) auto-mode-alist))
(setq auto-mode-alist (cons '("\.f$" . fortran-mode) auto-mode-alist))
(setq auto-mode-alist (cons '("\.F$" . fortran-mode) auto-mode-alist))
(setq fortran-do-indent 3)
(setq fortran-if-indent 3)

;;;
;;; to turn on color in fortran-mode (to set background color, see the 
;;; manpages for emacs or xemacs)
;;;

(add-hook 'fortran-mode-hook 'turn-on-font-lock)



