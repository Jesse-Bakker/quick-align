use epub::doc::EpubDoc;
use roxmltree::Document as XmlDocument;
use std::path::{Path, PathBuf};

#[derive(PartialEq, Eq, Debug)]
pub struct Paragraph {
    pub text: String,
}

#[derive(Debug)]
pub struct Chapter {
    pub paragraphs: Vec<Paragraph>,
}

pub struct Book {
    pub chapters: Vec<Chapter>,
}

#[derive(Debug, PartialEq)]
struct Link {
    path: PathBuf,
    id: Option<String>,
}

impl Link {
    fn from_href(href: &str) -> Self {
        let mut parts = href.splitn(2, '#');
        let filename = parts.next().unwrap();
        let id = parts.next();
        Self {
            path: filename.into(),
            id: id.map(|it| it.to_owned()),
        }
    }
}

fn all_text(node: roxmltree::Node) -> String {
    node.descendants()
        .filter_map(|node| if node.is_text() { node.text() } else { None })
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_paragraphs_from_text(
    text: &str,
    start_id: Option<&str>,
    end_id: Option<&str>,
) -> Vec<Paragraph> {
    let tree = XmlDocument::parse(text).unwrap();
    tree.root_element()
        .descendants()
        .skip_while(|node| start_id.is_some() && node.attribute("id") != start_id)
        .take_while(|node| end_id.is_none() || node.attribute("id") != end_id)
        .filter(|node| node.has_tag_name("p"))
        .map(|node| Paragraph {
            text: all_text(node),
        })
        .collect()
}

fn extract_chapter_paragraphs<R>(
    doc: &mut EpubDoc<R>,
    files: &Vec<String>,
    start_id: Option<&str>,
    end_id: Option<&str>,
) -> Vec<Paragraph>
where
    R: std::io::Read + std::io::Seek,
{
    let mut ret = vec![];
    for (idx, file) in files.iter().enumerate() {
        let contents = doc.get_resource_str(file).unwrap();
        let start_id = if idx == 0 { start_id } else { None };
        let end_id = if idx == files.len() - 1 { end_id } else { None };
        ret.extend(extract_paragraphs_from_text(&contents, start_id, end_id));
    }
    ret
}

struct EpubRange {
    resources: Vec<String>,
    start_id: Option<String>,
    end_id: Option<String>,
}

pub fn extract_chapters<P>(path: P) -> Book
where
    P: AsRef<Path>,
{
    let mut doc = EpubDoc::new(path).unwrap();
    let ranges = doc
        .toc
        .iter()
        .zip(doc.toc.iter().skip(1))
        .map(|(chapter, next_chapter)| {
            let chapter_link = Link::from_href(&chapter.content.to_string_lossy());
            let next_chapter_link = Link::from_href(&next_chapter.content.to_string_lossy());
            let start_idx = doc.resource_uri_to_chapter(&chapter_link.path).unwrap();
            let mut end_idx = doc
                .resource_uri_to_chapter(&next_chapter_link.path)
                .unwrap();
            if next_chapter_link.id.is_none() {
                end_idx -= 1;
            }
            let files = (start_idx..=end_idx)
                .map(|index| doc.spine[index].clone())
                .collect::<Vec<_>>();
            EpubRange {
                resources: files,
                start_id: chapter_link.id,
                end_id: next_chapter_link.id,
            }
        })
        .collect::<Vec<_>>();
    Book {
        chapters: ranges
            .iter()
            .map(|range| {
                extract_chapter_paragraphs(
                    &mut doc,
                    &range.resources,
                    range.start_id.as_deref(),
                    range.end_id.as_deref(),
                )
            })
            .map(|paragraphs| Chapter { paragraphs })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_from_href() {
        assert_eq!(
            Link::from_href("test/file.html"),
            Link {
                path: "test/file.html".into(),
                id: None
            }
        );

        assert_eq!(
            Link::from_href("test/file.html#id123"),
            Link {
                path: "test/file.html".into(),
                id: Some("id123".into()),
            }
        )
    }

    #[test]
    fn test_1() {
        extract_chapters("test2.epub");
        panic!();
    }

    #[test]
    fn test_extract_paragraphs_from_text_simple() {
        let xml = r#"
        <html>
            <body>
                <p>text</p>
                <p>more text</p>
                <p><em>Bold text</em> more</p>
            </body>
        </html>
"#;
        assert_eq!(
            extract_paragraphs_from_text(xml, None, None),
            vec![
                Paragraph {
                    text: "text".into()
                },
                Paragraph {
                    text: "more text".into()
                },
                Paragraph {
                    text: "Bold text more".into()
                }
            ]
        );
    }
}
