"""
Definitions and constructors for Fovea dialect operations.
"""
from typing import List, Dict, Any, Optional
from .ir import Operation, Value, Type, TensorType, F32Type

class FoveaDialect:
    """Builder methods for fovea dialect operations."""

    @staticmethod
    def black_level(
        raw_in: Value,
        offset: float,
        saturation: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, raw_in.type)
        return Operation(
            name="fovea.black_level",
            operands=[raw_in],
            results=[res],
            attributes={"black_level_offset": offset, "saturation_limit": saturation}
        )

    @staticmethod
    def lens_shading(
        raw_in: Value,
        poly_coefficients: List[float],
        optical_center: List[float],
        result_name: str
    ) -> Operation:
        res = Value(result_name, raw_in.type)
        return Operation(
            name="fovea.lens_shading",
            operands=[raw_in],
            results=[res],
            attributes={
                "radial_poly_coefficients": poly_coefficients,
                "optical_center": optical_center
            }
        )

    @staticmethod
    def defective_pixel(
        raw_in: Value,
        threshold: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, raw_in.type)
        return Operation(
            name="fovea.defective_pixel",
            operands=[raw_in],
            results=[res],
            attributes={"variance_threshold": threshold}
        )

    @staticmethod
    def demosaic(
        raw_in: Value,
        cfa_pattern: str,
        algorithm: str,
        result_name: str,
        rgb_shape: Optional[List[int]] = None
    ) -> Operation:
        shape = rgb_shape or (raw_in.type.shape + [3] if isinstance(raw_in.type, TensorType) else [-1, -1, 3])
        res = Value(result_name, TensorType(shape, F32Type()))
        return Operation(
            name="fovea.demosaic",
            operands=[raw_in],
            results=[res],
            attributes={"cfa_pattern": cfa_pattern, "algorithm": algorithm}
        )

    @staticmethod
    def white_balance(
        rgb_in: Value,
        gain_r: float,
        gain_g: float,
        gain_b: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, rgb_in.type)
        return Operation(
            name="fovea.white_balance",
            operands=[rgb_in],
            results=[res],
            attributes={"gain_r": gain_r, "gain_g": gain_g, "gain_b": gain_b}
        )

    @staticmethod
    def color_correction(
        rgb_in: Value,
        matrix_3x3: List[float],
        result_name: str
    ) -> Operation:
        res = Value(result_name, rgb_in.type)
        return Operation(
            name="fovea.color_correction",
            operands=[rgb_in],
            results=[res],
            attributes={"matrix_3x3": matrix_3x3}
        )

    @staticmethod
    def bilateral(
        rgb_in: Value,
        radius: int,
        sigma_spatial: float,
        sigma_range: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, rgb_in.type)
        return Operation(
            name="fovea.bilateral",
            operands=[rgb_in],
            results=[res],
            attributes={
                "radius": radius,
                "sigma_spatial": sigma_spatial,
                "sigma_range": sigma_range
            }
        )

    @staticmethod
    def guided_filter(
        guide_in: Value,
        input_in: Value,
        radius: int,
        eps: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, input_in.type)
        return Operation(
            name="fovea.guided_filter",
            operands=[guide_in, input_in],
            results=[res],
            attributes={"radius": radius, "eps": eps}
        )

    @staticmethod
    def unsharp_mask(
        rgb_in: Value,
        strength: float,
        threshold: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, rgb_in.type)
        return Operation(
            name="fovea.unsharp_mask",
            operands=[rgb_in],
            results=[res],
            attributes={"strength": strength, "threshold": threshold}
        )

    @staticmethod
    def hdr_fuse(
        under_exp: Value,
        mid_exp: Value,
        over_exp: Value,
        w_contrast: float,
        w_saturation: float,
        w_exposedness: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, mid_exp.type)
        return Operation(
            name="fovea.hdr_fuse",
            operands=[under_exp, mid_exp, over_exp],
            results=[res],
            attributes={
                "w_contrast": w_contrast,
                "w_saturation": w_saturation,
                "w_exposedness": w_exposedness
            }
        )

    @staticmethod
    def tone_map(
        rgb_in: Value,
        operator_type: str,
        key_value: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, rgb_in.type)
        return Operation(
            name="fovea.tone_map",
            operands=[rgb_in],
            results=[res],
            attributes={"operator_type": operator_type, "key_value": key_value}
        )

    @staticmethod
    def gamma(
        rgb_in: Value,
        gamma_value: float,
        result_name: str
    ) -> Operation:
        res = Value(result_name, rgb_in.type)
        return Operation(
            name="fovea.gamma",
            operands=[rgb_in],
            results=[res],
            attributes={"gamma_value": gamma_value}
        )
