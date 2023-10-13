### Problems

- Single token description: 
	- bug id 283676 description "SSIA" ; 
	- bug_id 303061 description "HEAD. "
	- bug_id 284297 description "see "
	- bug_id 347342 description "[]"
	- bug_id 271977 description "-"
	- even though sometimes necessary "Critical!!!"
- empty description
	- bug_id 9818 "description": " "
- (Non necessary tokens (duplicated \\n at the end of bug_id 342347))
### Preprocessing required
- remove empty
- remove low tokens numbers & visualize if change
- do the stemmization? YES as seen with the embedding