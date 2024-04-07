

function f32_x1024_dot(vx, vy) result(res) bind(c, name="f32_x1024_dot")
    implicit none
    real, dimension(1024), intent(in) :: vx
    real, dimension(1024), intent(in) :: vy
    real :: res

    res = DOT_PRODUCT(vx, vy)
end function f32_x1024_dot