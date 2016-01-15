;;; fix problems leading to doc++ failure
;;; underscores in doc comments are escaped with backslashes

;;; restrictions: no C-style comments _within_ a doc comment
;;; no TeX commands making special use of the backslash character

;;; run interactively, or use in a script as in the following examples:
;;; emacs -batch foo.h -l $HOME/linear_solvers/babel/docfix.el -f save-buffer
;;; or, in a bash shell,
;;; for i in *.h;do emacs -batch $i -l $HOME/linear_solvers/babel/docfix.el -f save-buffer; done                                                                                                        


(defun docfix ()
  "fixes doc comments (deliminated by /** and */) by escaping underscores"
  (interactive)
  (let ((doccom-start-tag "/**")
	(doccom-end-tag "*/")
	(doccom-start (point-min))
	(doccom-end (point-min))
	)
    (save-excursion
      (goto-char (point-min))
      (while doccom-start
	(setq doccom-start (search-forward doccom-start-tag nil t))
	(setq doccom-end (search-forward doccom-end-tag nil t))
	(if (and doccom-start doccom-end)
	    (save-restriction
	      (narrow-to-region doccom-start doccom-end)
	      (goto-char doccom-start)
	      (while (re-search-forward "[^\\]_" nil t)
		(backward-char)
		(insert-char ?\\ 1)
		)
	      )
	  )
	)
      )
    )
  )

(docfix)
