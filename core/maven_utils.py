"""
Utilities for parsing and analysing Maven artefact URLs.

This module exposes helpers to extract the so‑called GAV triplet
(groupId, artifactId and version) from a URL pointing at
``https://mvnrepository.com``.  It also implements logic to resolve
additional metadata via Maven Central where possible.  If the
requested piece of information cannot be found on Maven Central, the
original mvnrepository.com page will be consulted as a fall back.

The primary entry point is the :func:`analyze_maven_repository_url`
function which takes a single URL and returns a mapping containing
the extracted GAV together with license metadata.

Examples
========

>>> analyze_maven_repository_url(
...     "https://mvnrepository.com/artifact/org.slf4j/slf4j-reload4j/1.7.36"
... )
{'group_id': 'org.slf4j', 'artifact_id': 'slf4j-reload4j', 'version': '1.7.36',
 'license': 'Apache License 2.0', 'license_url': 'https://www.apache.org/licenses/LICENSE-2.0.txt',
 'copyright': 'Copyright 1999‑2024 The Apache Software Foundation'}

The above example illustrates the shape of the returned dictionary.  The
exact values depend on the artefact in question and the availability of
metadata on Maven Central.  When no licence can be found the
``license`` field will be set to ``None``.

Note
====

Network access is required to resolve metadata from Maven Central or
mvnrepository.com.  If network calls fail this module will still
attempt to extract as much information as possible from the URL
itself.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
from urllib.parse import urlparse, unquote
import logging

# requests is a commonly available third party HTTP client.  It
# provides better ergonomics than urllib.request and is used
# throughout this module for simplicity.  If requests is not
# available the code will fall back to urllib as a last resort.
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

# Import urllib for fallback
try:
    import urllib.request
    import urllib.error
except ImportError:
    urllib = None  # type: ignore


class MavenURLParseError(ValueError):
    """Raised when a URL cannot be interpreted as a mvnrepository.com URL."""


@dataclass
class GAV:
    """Simple container for Maven groupId/artifactId/version triplets."""

    group_id: str
    artifact_id: str
    version: Optional[str] = None

    def as_path(self) -> str:
        """Return a Maven Central path representation of the group and artifact.

        Example
        -------
        >>> GAV('org.slf4j', 'slf4j-api', '1.7.36').as_path()
        'org/slf4j/slf4j-api/1.7.36'
        """
        if self.version:
            return f"{self.group_id.replace('.', '/')}/{self.artifact_id}/{self.version}"
        return f"{self.group_id.replace('.', '/')}/{self.artifact_id}"


def _convert_maven_central_url_to_mvnrepository_format(url: str) -> Optional[str]:
    """将Maven Central URL转换为mvnrepository.com URL格式。
    
    支持的输入格式:
    - https://repo1.maven.org/maven2/group/path/artifact/version/artifact-version.jar
    - repo1.maven.org/maven2/group/path/artifact/version/artifact-version.jar
    
    转换为:
    - https://mvnrepository.com/artifact/group.path/artifact/version
    
    Parameters
    ----------
    url: str
        Maven Central URL
        
    Returns
    -------
    str or None
        转换后的mvnrepository.com URL，如果无法转换则返回None
    """
    logger = logging.getLogger("maven_utils.convert_url")
    logger.debug(f"Converting Maven Central URL: {url}")
    
    # 确保URL以https://开头，如果不是则添加
    if not url.startswith("http"):
        url = "https://" + url
    
    # 解析URL
    parsed = urlparse(url)
    
    # 检查是否是Maven Central URL
    if not parsed.netloc or "repo1.maven.org" not in parsed.netloc:
        logger.debug(f"URL is not a Maven Central URL: {url}")
        return None
    
    # 解析路径
    path = parsed.path.strip("/")
    if not path.startswith("maven2/"):
        logger.debug(f"URL path does not start with maven2/: {url}")
        return None
    
    # 移除maven2/前缀
    path = path[len("maven2/"):]
    parts = path.split("/")
    
    # 至少需要group路径、artifact、version
    if len(parts) < 3:
        logger.debug(f"URL path does not have enough components: {url}")
        return None
    
    # 提取version（最后一个部分）
    version = parts[-2]  # 倒数第二个是版本号
    artifact = parts[-3]  # 倒数第三个是artifactId
    
    # 提取groupId（前面的所有部分用点连接）
    group_parts = parts[:-3]  # 除了最后三个部分（artifact, version, filename）
    if not group_parts:
        logger.debug(f"Could not extract group from URL: {url}")
        return None
    
    group_id = ".".join(group_parts)
    
    # 构造mvnrepository.com URL
    mvn_url = f"https://mvnrepository.com/artifact/{group_id}/{artifact}/{version}"
    logger.debug(f"Converted to mvnrepository URL: {mvn_url}")
    
    return mvn_url


def _http_get(url: str) -> Tuple[Optional[str], Optional[int]]:
    """Best effort helper to perform a GET request and return the text body and status.

    This function will attempt to use the :mod:`requests` library if
    available.  If ``requests`` is not installed it will fall back to
    Python's built‑in urllib.  Both approaches return a tuple of
    ``(text, status_code)``.  If the request fails, ``(None, None)`` is
    returned.
    """
    logger = logging.getLogger("maven_utils.http")
    # Use a browser-like User-Agent to avoid simple bot-blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) LicenseDetector/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    # Simple retry loop for transient network issues or temporary blocks
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            logger.debug(f"HTTP GET (attempt {attempt}): {url}")
            if _HAS_REQUESTS and requests is not None:
                resp = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
                logger.debug(f"HTTP GET status: {resp.status_code} for {url}")
                logger.debug(f"Response headers: {dict(resp.headers)}")
                logger.debug(f"Response content length: {len(resp.text) if resp.text else 0}")
                
                # 检查内容类型
                content_type = resp.headers.get('content-type', '')
                logger.debug(f"Content-Type: {content_type}")
                
                # 如果是HTML内容，可能不是有效的POM文件
                if 'html' in content_type.lower():
                    logger.warning(f"Response appears to be HTML, not a POM file: {url}")
                    return resp.text, resp.status_code
                
                return resp.text, resp.status_code
            elif urllib is not None:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as fh:
                    charset = fh.headers.get_content_charset() or "utf-8"
                    body = fh.read().decode(charset, errors="replace")
                    status = fh.getcode()
                    logger.debug(f"HTTP GET status: {status} for {url}")
                    logger.debug(f"Response content length: {len(body)}")
                    return body, status
            else:
                # Neither requests nor urllib is available
                logger.error(f"No HTTP client available for request to {url}")
                return None, None
        except Exception as exc:
            logger.warning(f"HTTP GET attempt {attempt} failed for {url}: {exc}")
            if attempt == attempts:
                logger.error(f"HTTP GET ultimately failed for {url} after {attempts} attempts")
                return None, None
            # short backoff
            try:
                import time

                time.sleep(0.5 * attempt)
            except Exception:
                pass
    
    # 如果循环完成但没有返回，确保返回默认值
    return None, None


def parse_mvnrepository_url(url: str) -> GAV:
    """Parse a mvnrepository.com URL and return its GAV components.

    The expected URL forms are::

        https://mvnrepository.com/artifact/<group>/<artifact>
        https://mvnrepository.com/artifact/<group>/<artifact>/<version>

    The groupId may contain dots, the artifactId is assumed to be the
    second path segment and the optional version is the third.  Any
    query parameters or fragments will be ignored.

    Parameters
    ----------
    url: str
        A URL pointing at a mvnrepository.com artefact page.

    Returns
    -------
    :class:`GAV`
        The parsed groupId, artifactId and version (if present).

    Raises
    ------
    MavenURLParseError
        If the URL does not point at mvnrepository.com or does not follow
        the expected path structure.
    """
    logger = logging.getLogger("maven_utils.parse")
    logger.debug(f"Parsing mvnrepository URL: {url}")
    parsed = urlparse(url)
    if not parsed.netloc or "mvnrepository.com" not in parsed.netloc:
        raise MavenURLParseError(f"Unsupported host in URL: {url}")

    # Normalise path: remove leading/trailing slashes and decode percent encoding.
    path = unquote(parsed.path.strip("/"))
    parts = path.split("/")

    # Expected prefix is 'artifact', followed by at least group and artifact.
    if len(parts) < 3 or parts[0] != "artifact":
        raise MavenURLParseError(f"Unexpected path structure: {path}")

    group_id = parts[1]
    artifact_id: str = parts[2] if len(parts) > 2 else ""
    version: Optional[str] = parts[3] if len(parts) > 3 else None

    if not group_id or not artifact_id:
        raise MavenURLParseError(f"Missing group or artifact in URL: {url}")

    gav = GAV(group_id=group_id, artifact_id=artifact_id, version=version)
    logger.info(f"Parsed GAV: {gav}")
    return gav


def _parse_maven_metadata(xml_text: str) -> Optional[str]:
    """Extract the latest or release version from a maven‑metadata.xml.

    When analysing URLs without an explicit version the maven metadata
    file hosted on Maven Central can be used to discover the latest
    available version.  This helper prefers the ``<release>`` element
    inside ``<versioning>``.  If that is not present it falls back to
    ``<latest>`` or the last entry in the list of ``<version>`` elements.

    Parameters
    ----------
    xml_text: str
        The contents of a maven‑metadata.xml file.

    Returns
    -------
    str or None
        The chosen version string, or ``None`` if it cannot be
        determined.
    """
    try:
        root = ET.fromstring(xml_text)
        versioning = root.find("versioning")
        if versioning is not None:
            release = versioning.findtext("release")
            if release:
                return release.strip()
            latest = versioning.findtext("latest")
            if latest:
                return latest.strip()
            versions = versioning.find("versions")
            if versions is not None:
                version_list = [v.text for v in versions.findall("version") if v.text]
                if version_list:
                    return version_list[-1].strip()
        return None
    except ET.ParseError:
        return None


def resolve_latest_version(gav: GAV) -> Optional[str]:
    """Attempt to resolve the latest version of a given group/artifact.

    This helper fetches the ``maven‑metadata.xml`` from Maven Central
    and parses it to find a suitable version.  If network access fails
    or the metadata does not contain a version a ``None`` is returned.

    Parameters
    ----------
    gav: :class:`GAV`
        A GAV with ``version`` attribute set to ``None``.

    Returns
    -------
    str or None
        The resolved version string, or ``None``.
    """
    logger = logging.getLogger("maven_utils.resolve")
    metadata_url = (
        f"https://repo1.maven.org/maven2/{gav.group_id.replace('.', '/')}/"
        f"{gav.artifact_id}/maven-metadata.xml"
    )
    logger.debug(f"Resolving latest version using metadata URL: {metadata_url}")
    text, status = _http_get(metadata_url)
    if status and 200 <= status < 300 and text:
        ver = _parse_maven_metadata(text)
        logger.info(f"Resolved latest version for {gav.group_id}:{gav.artifact_id} -> {ver}")
        return ver
    logger.info(f"Could not resolve latest version for {gav.group_id}:{gav.artifact_id} (status={status})")
    return None


def _extract_license_from_pom(pom_xml: str) -> Tuple[List[Dict], Optional[str], Optional[str]]:
    """Parse a POM file and extract licence information.

    Returns a tuple ``(licenses_list, license_url, copyright)``.  Each
    element may be ``None`` if it is not present in the POM.  When
    multiple licences are declared only the first one is returned to
    simplify downstream processing.  Copyright strings are extracted
    heuristically by scanning for any line containing the word
    ``copyright`` (case–insensitive).
    """
    logger = logging.getLogger("maven_utils.pom")
    try:
        root = ET.fromstring(pom_xml)
    except ET.ParseError as exc:
        logger.warning(f"Failed to parse POM XML: {exc}")
        return [], None, None

    # licence extraction - 提取所有license
    licenses_list = []
    licenses = root.find("licenses")
    if licenses is not None:
        licence_elems = licenses.findall("license")
        for lic_elem in licence_elems:
            # Extract name and url if present
            name_text = lic_elem.findtext("name")
            url_text = lic_elem.findtext("url")
            if name_text or url_text:
                licenses_list.append({
                    "name": name_text.strip() if name_text else None,
                    "url": url_text.strip() if url_text else None
                })

    # copyright extraction: search in description or other top level text
    copyright_notice: Optional[str] = None
    # Flatten the entire XML to a string without tags to search for
    # copyright mentions.  While not perfect this method is
    # reasonably robust.
    text_content = ET.tostring(root, encoding="unicode", method="text")
    match = re.search(r"copyright[^\n]*", text_content, re.IGNORECASE)
    if match:
        copyright_notice = match.group(0).strip()

    logger.debug(f"Extracted from POM - licenses: {licenses_list}, copyright_notice: {copyright_notice}")
    return licenses_list, None, copyright_notice


def fetch_license_from_maven_central(gav: GAV) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    直接下载 POM 并解析，不再先尝试 search API。
    """
    logger = logging.getLogger("maven_utils.fetch")
    version = gav.version
    if not version:
        logger.debug(f"No version provided for {gav.group_id}:{gav.artifact_id}, attempting to resolve latest")
        version = resolve_latest_version(gav)
        if not version:
            logger.info(f"Unable to determine version for {gav.group_id}:{gav.artifact_id}")
            return None, None, None

    # 直接下载 POM 文件
    group_path = gav.group_id.replace('.', '/')
    pom_path = f"{group_path}/{gav.artifact_id}/{version}/{gav.artifact_id}-{version}.pom"
    # 使用 repo1.maven.org 而不是 search.maven.org/remotecontent
    pom_url = f"https://repo1.maven.org/maven2/{pom_path}"
    logger.debug(f"Fetching POM from: {pom_url}")
    pom_text, status = _http_get(pom_url)
    logger.debug(f"POM download result - status: {status}, content length: {len(pom_text) if pom_text else 0}")
    
    if status and 200 <= status < 300 and pom_text:
        # 检查是否是有效的XML内容
        if pom_text.strip().startswith('<?xml') or '<project' in pom_text:
            logger.debug("POM content appears to be valid XML")
        else:
            logger.warning("POM content does not appear to be valid XML")
            logger.debug(f"First 500 characters of content: {pom_text[:500]}")
            
        # 递归查找license信息
        licenses_list, _, cp = _extract_license_from_pom_recursive(pom_text, gav)
        logger.debug(f"Extracted licenses: {licenses_list}, copyright: {cp}")
        
        # 构造copyright notice，优先使用从POM中提取的copyright
        copyright_notice = cp if cp else _construct_copyright_notice(gav.group_id, None)
        logger.debug(f"Constructed copyright notice: {copyright_notice}")
        
        # 检查licenses_list是否包含实际的license信息
        if licenses_list and len(licenses_list) > 0 and (licenses_list[0]["name"] or licenses_list[0]["url"]):
            # 使用大模型将license信息转换为SPDX expression
            spdx_expression = _convert_licenses_to_spdx(licenses_list)
            logger.info(f"POM license for {gav.group_id}:{gav.artifact_id}:{version} -> {spdx_expression}")
            return spdx_expression, None, copyright_notice
        else:
            # 即使没有找到license，也要返回copyright信息
            logger.info(f"No licenses found for {gav.group_id}:{gav.artifact_id}:{version}, but returning copyright notice: {copyright_notice}")
            return None, None, copyright_notice
    logger.info(f"POM not available or failed for {gav.group_id}:{gav.artifact_id}:{version} (status={status})")
    return None, None, None


def _extract_license_from_pom_recursive(pom_xml: str, gav: GAV, max_depth: int = 5) -> Tuple[List[Dict], Optional[str], Optional[str]]:
    """
    递归提取POM中的license信息，直到找到license或达到最大深度。
    """
    logger = logging.getLogger("maven_utils.pom_recursive")
    
    if max_depth <= 0:
        logger.debug("Maximum recursion depth reached")
        return [], None, None
        
    try:
        root = ET.fromstring(pom_xml)
        logger.debug(f"Successfully parsed POM XML for {gav.group_id}:{gav.artifact_id}:{gav.version}")
    except ET.ParseError as exc:
        logger.warning(f"Failed to parse POM XML for {gav.group_id}:{gav.artifact_id}:{gav.version}: {exc}")
        return [], None, None

    # 处理XML命名空间
    namespaces = {'maven': 'http://maven.apache.org/POM/4.0.0'}
    
    # 提取当前POM的license信息
    licenses_list = []
    # 先尝试带命名空间的查找，如果找不到再尝试不带命名空间的查找
    licenses = root.find("maven:licenses", namespaces)
    if licenses is None:
        licenses = root.find("licenses")
        
    if licenses is not None:
        licence_elems = licenses.findall("maven:license", namespaces)
        if not licence_elems:
            licence_elems = licenses.findall("license")
            
        logger.debug(f"Found {len(licence_elems)} license elements in {gav.group_id}:{gav.artifact_id}:{gav.version}")
        for lic_elem in licence_elems:
            # 同样处理命名空间 - 正确的方式
            name_elem = lic_elem.find("maven:name", namespaces)
            if name_elem is None:
                name_elem = lic_elem.find("name")
                
            url_elem = lic_elem.find("maven:url", namespaces)
            if url_elem is None:
                url_elem = lic_elem.find("url")
                
            name_text = name_elem.text if name_elem is not None else None
            url_text = url_elem.text if url_elem is not None else None
            
            logger.debug(f"License element - name: {name_text}, url: {url_text}")
            if name_text or url_text:
                licenses_list.append({
                    "name": name_text.strip() if name_text else None,
                    "url": url_text.strip() if url_text else None
                })

    # copyright extraction
    copyright_notice: Optional[str] = None
    text_content = ET.tostring(root, encoding="unicode", method="text")
    match = re.search(r"copyright[^\n]*", text_content, re.IGNORECASE)
    if match:
        copyright_notice = match.group(0).strip()

    logger.debug(f"Current POM {gav.group_id}:{gav.artifact_id}:{gav.version} - Licenses found: {len(licenses_list)}, Copyright: {copyright_notice}")

    # 如果当前POM有license信息，直接返回
    if licenses_list and len(licenses_list) > 0:
        logger.debug(f"Found licenses in current POM: {licenses_list}")
        return licenses_list, None, copyright_notice

    # 如果当前POM没有license，尝试从parent POM获取
    # 先尝试带命名空间的查找，如果找不到再尝试不带命名空间的查找
    parent = root.find("maven:parent", namespaces)
    if parent is None:
        parent = root.find("parent")
        
    if parent is not None:
        # 同样处理命名空间 - 正确的方式
        group_elem = parent.find("maven:groupId", namespaces)
        if group_elem is None:
            group_elem = parent.find("groupId")
            
        artifact_elem = parent.find("maven:artifactId", namespaces)
        if artifact_elem is None:
            artifact_elem = parent.find("artifactId")
            
        version_elem = parent.find("maven:version", namespaces)
        if version_elem is None:
            version_elem = parent.find("version")
        
        parent_group_id = group_elem.text if group_elem is not None else None
        parent_artifact_id = artifact_elem.text if artifact_elem is not None else None
        parent_version = version_elem.text if version_elem is not None else None
        
        logger.debug(f"Parent POM info - Group: {parent_group_id}, Artifact: {parent_artifact_id}, Version: {parent_version}")
        
        if parent_group_id and parent_artifact_id and parent_version:
            # 构造parent POM的URL并下载
            parent_group_path = parent_group_id.replace('.', '/')
            parent_pom_path = f"{parent_group_path}/{parent_artifact_id}/{parent_version}/{parent_artifact_id}-{parent_version}.pom"
            parent_pom_url = f"https://repo1.maven.org/maven2/{parent_pom_path}"
            logger.debug(f"Fetching parent POM from: {parent_pom_url}")
            parent_pom_text, parent_status = _http_get(parent_pom_url)
            logger.debug(f"Parent POM download result - status: {parent_status}, content length: {len(parent_pom_text) if parent_pom_text else 0}")
            
            if parent_status and 200 <= parent_status < 300 and parent_pom_text:
                # 检查parent POM内容
                if parent_pom_text.strip().startswith('<?xml') or '<project' in parent_pom_text:
                    logger.debug("Parent POM content appears to be valid XML")
                else:
                    logger.warning("Parent POM content does not appear to be valid XML")
                    logger.debug(f"First 500 characters of parent POM content: {parent_pom_text[:500]}")
                    
                parent_gav = GAV(group_id=parent_group_id, artifact_id=parent_artifact_id, version=parent_version)
                logger.debug(f"Recursively calling _extract_license_from_pom_recursive for parent POM")
                # 修复：正确处理递归调用的返回值
                parent_licenses, parent_url, parent_copyright = _extract_license_from_pom_recursive(parent_pom_text, parent_gav, max_depth - 1)
                # 如果parent POM找到了license信息，返回parent的结果
                if parent_licenses and len(parent_licenses) > 0:
                    logger.debug(f"Found licenses in parent POM: {parent_licenses}")
                    # 如果当前POM没有copyright但parent有，使用parent的copyright
                    if not copyright_notice and parent_copyright:
                        copyright_notice = parent_copyright
                    return parent_licenses, parent_url, copyright_notice
                else:
                    # 如果parent也没有license，但可能有copyright，合并copyright信息
                    if not copyright_notice and parent_copyright:
                        copyright_notice = parent_copyright
                    logger.debug("Parent POM also has no licenses, returning current results with possible copyright")
            else:
                logger.warning(f"Failed to fetch parent POM from {parent_pom_url}, status: {parent_status}")
        else:
            logger.warning(f"Incomplete parent POM information: group_id={parent_group_id}, artifact_id={parent_artifact_id}, version={parent_version}")
    else:
        logger.debug("No parent POM found in current POM")
    
    # 返回当前的结果（可能包含从parent获取的copyright）
    logger.debug(f"Returning results - Licenses: {licenses_list}, Copyright: {copyright_notice}")
    return licenses_list, None, copyright_notice


def _convert_licenses_to_spdx(licenses_list: List[Dict]) -> Optional[str]:
    """
    使用大模型将license信息转换为SPDX expression。
    """
    if not licenses_list:
        return None
        
    logger = logging.getLogger("maven_utils.spdx")
    
    try:
        # 导入必要的模块
        from core.llm_provider import get_llm_provider
        from core.config import LLM_CONFIG
        import yaml
        import os
        import json
        import re
        
        # 加载prompt
        with open("prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
            
        # 构造prompt
        licenses_str = "\n".join([f"- Name: {lic['name'] or 'N/A'}, URL: {lic['url'] or 'N/A'}" for lic in licenses_list])
        prompt = prompts["license_standardize"].format(license_string=licenses_str)
        
        logger.info("License Standardization Request:")
        logger.info(f"Prompt: {prompt}")
        
        # 调用大模型
        provider = get_llm_provider()
        try:
            response = provider.generate(prompt)
        except RuntimeError as e:
            if "This event loop is already running" in str(e):
                logger.warning("Event loop is already running, skipping LLM-based license standardization")
                # 如果大模型调用失败，直接返回第一个license的名称
                if licenses_list and licenses_list[0]["name"]:
                    return licenses_list[0]["name"]
                return None
            else:
                # 重新抛出其他RuntimeError异常
                raise e
        
        logger.info("License Standardization Response:")
        logger.info(f"Response: {response}")
        
        if response:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                spdx_identifier = result.get("spdx_identifier")
                if spdx_identifier and spdx_identifier != "UNKNOWN":
                    return spdx_identifier
                    
    except Exception as e:
        logger.error(f"Failed to convert licenses to SPDX: {str(e)}", exc_info=True)
        
    # 如果大模型转换失败，返回第一个license的名称
    if licenses_list and licenses_list[0]["name"]:
        return licenses_list[0]["name"]
        
    return None


def _construct_copyright_notice(group_id: str, existing_copyright: Optional[str]) -> str:
    """
    构造copyright notice。
    """
    if existing_copyright:
        return existing_copyright
        
    # 从group_id提取组织名
    org_name = group_id
    if '.' in group_id:
        # 对于org.slf4j这样的group_id，我们想要提取"Slf4j"作为组织名
        parts = group_id.split('.')
        if len(parts) >= 2:
            # 取第二部分作为组织名，如slf4j -> Slf4j
            org_name = parts[1]
        else:
            # 如果只有一部分，就用第一部分
            org_name = parts[0]
    
    # 首字母大写
    org_name = org_name.capitalize()
    
    from datetime import datetime
    current_year = datetime.now().year
    
    return f"Copyright (c) {current_year} {org_name}"


def _fallback_extract_license_from_html(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    已废弃：不再从 mvnrepository.com HTML 页面抓取 license，直接返回 None。
    """
    logging.getLogger("maven_utils.fallback").info(f"Fallback HTML extraction skipped for {url}")
    return None, None, None


def analyze_maven_repository_url(url: str) -> Dict[str, Any]:
    """Analyse a single mvnrepository.com URL and return metadata.

    This high level helper ties together the various parsing and
    extraction routines.  It will always return a dictionary with the
    keys ``group_id`` and ``artifact_id``.  When a version can be
    determined it will be included under the ``version`` key.  If
    licence information is available the keys ``license``,
    ``license_url`` and ``copyright`` will be present as well.  Any
    value may be ``None`` if not resolvable.

    The resolution strategy proceeds as follows:

    1. Parse the URL to obtain the GAV.
    2. Attempt to resolve licence metadata via Maven Central by
       downloading and parsing the artefact's POM.
    3. If no licence is discovered fall back to scraping the
       mvnrepository.com page.

    Parameters
    ----------
    url: str
        A URL pointing at a page on mvnrepository.com representing a
        Maven artefact.

    Returns
    -------
    dict
        A dictionary of extracted metadata.  The keys ``license``,
        ``license_url`` and ``copyright`` will be omitted if no
        corresponding value can be found.

    Raises
    ------
    MavenURLParseError
        If the URL cannot be parsed into a GAV.
    """
    logger = logging.getLogger("maven_utils.analyze")
    logger.info(f"Analyze Maven URL: {url}")
    
    # 检查是否是Maven Central URL，如果是则转换为mvnrepository.com格式
    if "repo1.maven.org" in url:
        converted_url = _convert_maven_central_url_to_mvnrepository_format(url)
        if converted_url:
            logger.info(f"Converted Maven Central URL to: {converted_url}")
            url = converted_url
        else:
            logger.warning(f"Failed to convert Maven Central URL: {url}")
    
    gav = parse_mvnrepository_url(url)
    result: Dict[str, Any] = {
        "group_id": gav.group_id,
        "artifact_id": gav.artifact_id,
    }
    if gav.version:
        result["version"] = gav.version
    else:
        # Try to determine a version from Maven Central metadata
        resolved_version = resolve_latest_version(gav)
        if resolved_version:
            result["version"] = resolved_version

    # Try to fetch licence info from Maven Central
    lic_name, lic_url, cp_notice = fetch_license_from_maven_central(gav)
    if lic_name:
        result["license"] = lic_name
        result["license_url"] = lic_url
        result["license_source"] = "maven_central"
    else:
        # ensure license_url/copyright are not left undefined here
        if lic_url:
            result["license_url"] = lic_url
        if cp_notice:
            result["copyright"] = cp_notice

    # Fall back to scraping mvnrepository if no licence found
    if not lic_name:
        logger.debug("No license found in POM, attempting fallback HTML extraction")
        fb_name, fb_url, fb_cp = _fallback_extract_license_from_html(url)
        if fb_name and "license" not in result:
            result["license"] = fb_name
            result["license_source"] = "mvnrepository"
        if fb_url and "license_url" not in result:
            result["license_url"] = fb_url
        if fb_cp and "copyright" not in result:
            result["copyright"] = fb_cp

    # ensure license_source exists (None if not found)
    if "license_source" not in result:
        result["license_source"] = None

    logger.info(f"analyze_maven_repository_url result for {url}: {result}")

    return result


# For backwards compatibility some projects may import ``analyse_maven_repository_url``.
# Provide an alias to avoid breaking existing users.
def analyse_maven_repository_url(url: str) -> Dict[str, Any]:  # pragma: no cover
    return analyze_maven_repository_url(url)
