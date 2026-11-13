/* Test entry points.  */
#ifndef TESTS_H
#define TESTS_H 1

void test_strlen (void);
void test_strnlen (void);
void test_strchr (void);
void test_strchrnul (void);
void test_strrchr (void);

void test_strcmp (void);
void test_strncmp (void);
void test_strcasecmp (void);
void test_strncasecmp (void);

void test_strcpy (void);
void test_strncpy (void);
void test_strcat (void);
void test_strncat (void);

void test_strspn (void);
void test_strcspn (void);
void test_strpbrk (void);
void test_strstr (void);

void test_memcpy (void);
void test_mempcpy (void);
void test_memmove (void);
void test_memccpy (void);

void test_memchr (void);
void test_memrchr (void);
void test_rawmemchr (void);
void test_memcmp (void);
void test_memset (void);

/* exhaustive small-alphabet cases */
void exh_strstr (void);
void exh_span (int fk);
void exh_cmp (int fk, int is_n);
void exh_chr (int fk);

/* page-boundary / guard-page cases, folded into each function's result */
void edge_strlen (void);
void edge_strnlen (void);
void edge_strchr (void);
void edge_strchrnul (void);
void edge_strrchr (void);
void edge_strcmp (void);
void edge_strncmp (void);
void edge_strcasecmp (void);
void edge_strncasecmp (void);
void edge_strcpy (void);
void edge_strncpy (void);
void edge_strcat (void);
void edge_strncat (void);
void edge_strspn (void);
void edge_strcspn (void);
void edge_strpbrk (void);
void edge_strstr (void);
void edge_memcpy (void);
void edge_mempcpy (void);
void edge_memccpy (void);
void edge_memmove (void);
void edge_memchr (void);
void edge_memrchr (void);
void edge_rawmemchr (void);
void edge_memcmp (void);
void edge_memset (void);

#endif
