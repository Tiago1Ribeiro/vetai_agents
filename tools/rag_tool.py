"""
Processador de imagens para análise veterinária.

Funcionalidades:
- Validação de formatos
- Redimensionamento inteligente
- Conversão para base64
- Melhorias de imagem (contraste, brilho)
- Extração de metadados
"""

import base64
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
import hashlib


@dataclass
class ImageInfo:
    """Informação sobre uma imagem processada"""
    path: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    format: str
    mime_type: str
    file_size_kb: float
    base64_size_kb: float
    hash: str
    exif: Optional[dict] = None


class ImageProcessor:
    """
    Processador de imagens otimizado para VLMs.
    
    Prepara imagens para envio a APIs de visão, garantindo:
    - Tamanho adequado (max 1120x1120 para Llama Vision)
    - Formato compatível (JPEG/PNG)
    - Compressão eficiente
    """
    
    # Formatos suportados
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
    
    # Limites para diferentes VLMs
    VLM_LIMITS = {
        'llama-vision': {'max_size': 1120, 'max_file_mb': 20},
        'gemini': {'max_size': 2048, 'max_file_mb': 20},
        'default': {'max_size': 1120, 'max_file_mb': 10}
    }
    
    def __init__(
        self,
        target_vlm: str = 'default',
        quality: int = 85,
        enhance_images: bool = True
    ):
        """
        Args:
            target_vlm: VLM alvo ('llama-vision', 'gemini', 'default')
            quality: Qualidade JPEG (1-100)
            enhance_images: Se deve aplicar melhorias automáticas
        """
        self.limits = self.VLM_LIMITS.get(target_vlm, self.VLM_LIMITS['default'])
        self.quality = quality
        self.enhance_images = enhance_images
        self.max_size = self.limits['max_size']
    
    def validate_image(self, image_path: Union[str, Path]) -> tuple[bool, str]:
        """
        Valida se uma imagem é suportada.
        
        Returns:
            (is_valid, message)
        """
        path = Path(image_path)
        
        # Verificar se existe
        if not path.exists():
            return False, f"Ficheiro não encontrado: {path}"
        
        # Verificar extensão
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return False, f"Formato não suportado: {path.suffix}. Use: {self.SUPPORTED_FORMATS}"
        
        # Verificar tamanho do ficheiro
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.limits['max_file_mb']:
            return False, f"Ficheiro muito grande: {file_size_mb:.1f}MB (max: {self.limits['max_file_mb']}MB)"
        
        # Tentar abrir a imagem
        try:
            with Image.open(path) as img:
                img.verify()
            return True, "Imagem válida"
        except Exception as e:
            return False, f"Imagem corrompida ou inválida: {e}"
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Carrega uma imagem e converte para RGB se necessário"""
        img = Image.open(image_path)
        
        # Converter para RGB (remove alpha channel se existir)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Criar background branco para transparência
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    def resize_image(
        self,
        img: Image.Image,
        max_size: Optional[int] = None
    ) -> Image.Image:
        """
        Redimensiona imagem mantendo aspect ratio.
        
        Args:
            img: Imagem PIL
            max_size: Tamanho máximo (largura ou altura)
        """
        max_size = max_size or self.max_size
        
        # Só redimensionar se necessário
        if img.width <= max_size and img.height <= max_size:
            return img
        
        # Calcular novo tamanho mantendo aspect ratio
        ratio = min(max_size / img.width, max_size / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        
        # Usar LANCZOS para melhor qualidade
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    def enhance_image(self, img: Image.Image) -> Image.Image:
        """
        Aplica melhorias automáticas para melhor análise.
        
        Útil para imagens veterinárias que podem estar:
        - Mal iluminadas
        - Com pouco contraste
        - Ligeiramente desfocadas
        """
        # Melhorar contraste ligeiramente
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Melhorar nitidez
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        
        # Ajustar brilho se muito escura
        enhancer = ImageEnhance.Brightness(img)
        # Calcular brilho médio
        grayscale = img.convert('L')
        avg_brightness = sum(grayscale.getdata()) / len(list(grayscale.getdata()))
        
        if avg_brightness < 80:  # Imagem escura
            img = enhancer.enhance(1.2)
        elif avg_brightness > 200:  # Imagem muito clara
            img = enhancer.enhance(0.9)
        
        return img
    
    def extract_exif(self, img: Image.Image) -> Optional[dict]:
        """Extrai metadados EXIF da imagem"""
        try:
            exif = img._getexif()
            if exif:
                return {
                    ExifTags.TAGS.get(k, k): v
                    for k, v in exif.items()
                    if isinstance(v, (str, int, float, bytes))
                }
        except (AttributeError, KeyError):
            pass
        return None
    
    def to_base64(
        self,
        img: Image.Image,
        format: str = 'JPEG'
    ) -> tuple[str, str]:
        """
        Converte imagem para base64.
        
        Returns:
            (base64_string, mime_type)
        """
        buffer = io.BytesIO()
        
        # Guardar no buffer
        save_kwargs = {'quality': self.quality} if format == 'JPEG' else {}
        img.save(buffer, format=format, **save_kwargs)
        
        # Converter para base64
        b64 = base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
        mime = f"image/{format.lower()}"
        
        return b64, mime
    
    def compute_hash(self, img: Image.Image) -> str:
        """Calcula hash da imagem para deduplicação"""
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return hashlib.md5(buffer.getvalue()).hexdigest()[:16]
    
    def process_image(
        self,
        image_path: Union[str, Path],
        return_pil: bool = False
    ) -> Union[ImageInfo, tuple[ImageInfo, Image.Image]]:
        """
        Processa uma imagem completamente.
        
        Args:
            image_path: Caminho para a imagem
            return_pil: Se deve retornar também o objeto PIL
            
        Returns:
            ImageInfo ou (ImageInfo, PIL.Image)
        """
        path = Path(image_path)
        
        # Validar
        is_valid, msg = self.validate_image(path)
        if not is_valid:
            raise ValueError(msg)
        
        # Carregar
        img = self.load_image(path)
        original_size = img.size
        
        # Extrair EXIF antes de processar
        exif = self.extract_exif(img)
        
        # Redimensionar
        img = self.resize_image(img)
        
        # Melhorar (se ativado)
        if self.enhance_images:
            img = self.enhance_image(img)
        
        # Converter para base64
        b64, mime = self.to_base64(img)
        
        # Criar info
        info = ImageInfo(
            path=str(path),
            original_size=original_size,
            processed_size=img.size,
            format=path.suffix.upper().replace('.', ''),
            mime_type=mime,
            file_size_kb=path.stat().st_size / 1024,
            base64_size_kb=len(b64) / 1024,
            hash=self.compute_hash(img),
            exif=exif
        )
        
        if return_pil:
            return info, img
        return info
    
    def process_multiple(
        self,
        image_paths: list[Union[str, Path]],
        deduplicate: bool = True
    ) -> list[dict]:
        """
        Processa múltiplas imagens.
        
        Args:
            image_paths: Lista de caminhos
            deduplicate: Remove imagens duplicadas (por hash)
            
        Returns:
            Lista de dicts com {info, base64, mime_type}
        """
        results = []
        seen_hashes = set()
        
        for path in image_paths:
            try:
                info, img = self.process_image(path, return_pil=True)
                
                # Deduplicação
                if deduplicate and info.hash in seen_hashes:
                    print(f"⚠️ Imagem duplicada ignorada: {path}")
                    continue
                seen_hashes.add(info.hash)
                
                # Obter base64
                b64, mime = self.to_base64(img)
                
                results.append({
                    'info': info,
                    'base64': b64,
                    'mime_type': mime,
                    'data_url': f"data:{mime};base64,{b64}"
                })
                
            except Exception as e:
                print(f"❌ Erro ao processar {path}: {e}")
                continue
        
        return results
    
    def prepare_for_vlm(
        self,
        image_paths: list[Union[str, Path]],
        format: str = 'openai'
    ) -> list[dict]:
        """
        Prepara imagens no formato específico para VLMs.
        
        Args:
            image_paths: Lista de caminhos
            format: 'openai', 'anthropic', 'gemini'
            
        Returns:
            Lista de content blocks prontos para API
        """
        processed = self.process_multiple(image_paths)
        
        if format == 'openai':
            # Formato OpenAI/Together AI
            return [
                {
                    "type": "image_url",
                    "image_url": {"url": p['data_url']}
                }
                for p in processed
            ]
        
        elif format == 'anthropic':
            # Formato Anthropic
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": p['mime_type'],
                        "data": p['base64']
                    }
                }
                for p in processed
            ]
        
        elif format == 'gemini':
            # Formato Google Gemini (retorna PIL Images)
            return [
                Image.open(io.BytesIO(base64.b64decode(p['base64'])))
                for p in processed
            ]
        
        else:
            raise ValueError(f"Formato não suportado: {format}")
    
    @staticmethod
    def create_thumbnail(
        image_path: Union[str, Path],
        size: tuple[int, int] = (150, 150),
        output_path: Optional[str] = None
    ) -> str:
        """
        Cria thumbnail de uma imagem.
        
        Args:
            image_path: Caminho da imagem original
            size: Tamanho do thumbnail (largura, altura)
            output_path: Caminho de output (opcional)
            
        Returns:
            Caminho do thumbnail
        """
        path = Path(image_path)
        
        with Image.open(path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            if output_path is None:
                output_path = path.parent / f"{path.stem}_thumb{path.suffix}"
            
            img.save(output_path)
        
        return str(output_path)


# Funções utilitárias standalone
def quick_process(image_path: str) -> dict:
    """Processamento rápido de uma imagem"""
    processor = ImageProcessor()
    info = processor.process_image(image_path)
    _, img = processor.process_image(image_path, return_pil=True)
    b64, mime = processor.to_base64(img)
    
    return {
        'info': info,
        'base64': b64,
        'mime_type': mime,
        'data_url': f"data:{mime};base64,{b64}"
    }


def validate_images(image_paths: list[str]) -> dict:
    """Valida múltiplas imagens"""
    processor = ImageProcessor()
    results = {'valid': [], 'invalid': []}
    
    for path in image_paths:
        is_valid, msg = processor.validate_image(path)
        if is_valid:
            results['valid'].append(path)
        else:
            results['invalid'].append({'path': path, 'error': msg})
    
    return results

