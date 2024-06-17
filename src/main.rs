use std::fs::File;
use std::io::Read;
use winit::application::ApplicationHandler;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowAttributes};

use pixels::{Pixels, SurfaceTexture};

enum KeyPressState {
    Idle,
    Waiting(u8),
    Set(u8, u8),
}

struct Emulator {
    key_input: KeyPressState,
    // program: Vec<u16>,
    core: Core,
    window: Option<Window>,
    pixels: Option<Pixels>,
}

macro_rules! reg_x {
    ($opcode:expr) => {{
        (($opcode & BITMASK_REG_X) >> 8) as u8
    }};
}

macro_rules! reg_y {
    ($opcode:expr) => {{
        (($opcode & BITMASK_REG_Y) >> 4) as u8
    }};
}

fn draw(core: &Core, screen: &mut [u8]) {
    for (c, pix) in core.display.iter().zip(screen.chunks_exact_mut(4)) {
        let color = if *c {
            [0xff, 0xff, 0xff, 0xff]
        } else {
            [0, 0, 0, 0xff]
        };
        pix.copy_from_slice(&color);
    }
}

struct Core {
    ram: [u8; 0xFFF],
    registers: [u8; 16],
    dt: u8,
    st: u8,
    ix: u16,
    pc: u16,
    keys: u16,
    display: Vec<bool>,
    stack: Vec<u16>,
    width: u8,
    height: u8,
}

const BITMASK_REG_X: u16 = 0x0F00;
const BITMASK_REG_Y: u16 = 0x00F0;
const BITMASK_N: u16 = 0x000F;
const BITMASK_NN: u16 = 0x00FF;
const BITMASK_NNN: u16 = 0x0FFF;

impl Core {
    fn new(display_width: u8, display_height: u8, program: &Vec<u8>) -> Core {
        let mut ram = [0; 0xfff];
        let fonts: [u8; 80] = [
            0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
            0x20, 0x60, 0x20, 0x20, 0x70, // 1
            0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
            0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
            0x90, 0x90, 0xF0, 0x10, 0x10, // 4
            0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
            0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
            0xF0, 0x10, 0x20, 0x40, 0x40, // 7
            0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
            0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
            0xF0, 0x90, 0xF0, 0x90, 0x90, // A
            0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
            0xF0, 0x80, 0x80, 0x80, 0xF0, // C
            0xE0, 0x90, 0x90, 0x90, 0xE0, // D
            0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
            0xF0, 0x80, 0xF0, 0x80, 0x80, // F
        ];

        ram[..80].copy_from_slice(&fonts);
        for (i, instr) in program.iter().enumerate() {
            ram[0x0200 + i] = *instr;
        }
        // for i in program.
        // }
        Core {
            ram,
            registers: [0; 16],
            dt: 0,
            st: 0,
            ix: 0,
            pc: 0x200,
            keys: 0,
            display: vec![false; display_width as usize * display_height as usize],
            stack: Vec::new(),
            width: display_width,
            height: display_height,
        }
    }

    fn set_pixel(&mut self, x: u8, y: u8, set: bool) {
        self.display[(self.width as u16 * x as u16 + y as u16) as usize] = set;
    }
    fn get_pixel(&self, x: u8, y: u8) -> bool {
        self.display[(self.width as u16 * x as u16 + y as u16) as usize]
    }

    // 00E0
    fn cls(&mut self) {
        self.display = vec![false; self.width as usize * self.width as usize];
    }

    // 00EE
    // Return from a subroutine. Pops the value at the top of the stack (indicated by the stack pointer SP) and puts it in PC.
    // PC := stack[SP]
    fn ret(&mut self) {
        self.pc = self.stack.pop().expect("stack is empty")
    }

    // 1NNN
    fn jmp(&mut self, n: u16) {
        self.pc = n;
    }

    // CALL NNN — 2NNN
    fn call(&mut self, addr: u16) {
        self.stack.push(self.pc);
        self.pc = addr;
    }

    // SE VX, NN — 3XNN
    fn se_n(&mut self, vx: u8, n: u8) {
        if self.registers[vx as usize] == n {
            self.pc += 2;
        }
    }

    // SNE VX, NN — 4XNN
    fn sne(&mut self, vx: u8, n: u8) {
        if self.registers[vx as usize] != n {
            self.pc += 2;
        }
    }

    // SE VX, VY — 5XY0
    fn se(&mut self, vx: u8, vy: u8) {
        if self.registers[vx as usize] == self.registers[vy as usize] {
            self.pc += 2;
        }
    }

    // LD VX, NN — 6XNN
    fn ld(&mut self, vx: u8, n: u8) {
        self.registers[vx as usize] = n;
    }

    // ADD VX, NN — 7XNN
    fn add_n(&mut self, vx: u8, n: u8) {
        let new_register = self.registers[vx as usize].checked_add(n);
        match new_register {
            None => {
                self.registers[vx as usize] =
                    ((self.registers[vx as usize] as u16 + n as u16) & 0xFF) as u8;
                self.registers[15] = 1;
            }
            Some(v) => {
                self.registers[15] = 0;
                self.registers[vx as usize] = v;
            }
        }
    }

    // LD VX, VY — 8XY0
    fn ld_vx_vy(&mut self, vx: u8, vy: u8) {
        self.registers[vx as usize] += self.registers[vy as usize];
    }

    // OR VX, VY — 8XY1
    fn or(&mut self, vx: u8, vy: u8) {
        self.registers[vx as usize] |= self.registers[vy as usize];
    }

    // AND VX, VY — 8XY2
    fn and(&mut self, vx: u8, vy: u8) {
        self.registers[vx as usize] &= self.registers[vy as usize];
    }

    // XOR VX, VY — 8XY3
    fn xor(&mut self, vx: u8, vy: u8) {
        self.registers[vx as usize] ^= self.registers[vy as usize];
    }

    // ADD VX, VY — 8XY4
    fn add_vx_vy(&mut self, vx: u8, vy: u8) {
        let vxp: u16 = self.registers[vx as usize] as u16;
        let vyp: u16 = self.registers[vy as usize] as u16;
        if vxp + vyp > 0xFF {
            self.registers[15] = 1
        } else {
            self.registers[15] = 0
        }
        self.registers[vx as usize] = ((vxp + vyp) & 0xff) as u8;
    }

    // SUB VX, VY — 8XY5
    fn sub(&mut self, vx: u8, vy: u8) {
        let vxp: u16 = self.registers[vx as usize] as u16;
        let vyp: u16 = self.registers[vy as usize] as u16;
        if vxp > vyp {
            self.registers[15] = 1
        } else {
            self.registers[15] = 0
        }
        self.registers[vx as usize] = ((vyp - vxp) & 0xff) as u8;
    }

    // SHR VX {, VY} — 8XY6
    fn shr(&mut self, vx: u8) {
        self.registers[15] = self.registers[vx as usize] & 0x01;
        self.registers[vx as usize] >>= 1;
    }

    // SUBN VX, VY — 8XY7
    fn subn(&mut self, vx: u8, vy: u8) {
        let vxp: u16 = self.registers[vx as usize] as u16;
        let vyp: u16 = self.registers[vy as usize] as u16;
        if vyp > vxp {
            self.registers[15] = 1
        } else {
            self.registers[15] = 0
        }
        self.registers[vx as usize] = self.registers[vy as usize] - self.registers[vx as usize];
    }

    // SHL VX {, VY} — 8XYE
    fn shl(&mut self, vx: u8) {
        self.registers[15] = self.registers[vx as usize] & 0x80;
        self.registers[vx as usize] <<= 1;
    }

    // SNE VX, VY — 9XY0
    fn sne_vx_vy(&mut self, vx: u8, vy: u8) {
        if self.registers[vx as usize] != self.registers[vy as usize] {
            self.pc += 2;
        }
    }

    // LD I, NNN — ANNN
    fn ld_i(&mut self, n: u16) {
        self.ix = n;
    }

    // JMP V0, NNN — BNNN
    fn jmp_v0(&mut self, nnn: u16) {
        self.pc = self.registers[0] as u16 + nnn;
    }

    // RND VX, NN – CXNN
    fn rnd(&mut self, vx: u8, n: u8) {
        let rng: u8 = rand::random();
        self.registers[vx as usize] = rng & n;
    }

    // DRW VX, VY, N — DXYN
    fn drw(&mut self, vx: u8, vy: u8, n: u8) {
        /*
        Display n-byte sprite starting at memory location I at (Vx, Vy),
        set VF = collision.

        Draws a sprite at coordinate (VX, VY) that has a width of 8 pixels
        and a height of N pixels. Each row of 8 pixels is read as bit-coded
        (with the most significant bit of each byte displayed on the left)
        starting from memory location I; I value doesn't change after the
        execution of this instruction. As described above, VF is set to 1
        if any screen pixels are flipped from set to unset when the sprite
        is drawn, and to 0 if that doesn't happen.
         */
        let xcoord = self.registers[vx as usize];
        let ycoord = self.registers[vy as usize];
        let mut new_vf = 0;

        for row in 0..n {
            let cell = self.ram[(self.ix + row as u16) as usize];
            for col in 0..8 {
                if cell & (0x01 << (7 - col)) == 0 {
                    continue;
                }
                let curr_px = self.get_pixel(ycoord + row, xcoord + col);
                self.set_pixel(ycoord + row, xcoord + col, !curr_px);
                if curr_px {
                    new_vf = 1
                }
            }
        }
        self.registers[15] = new_vf;
    }

    // SKP VX — EX9E
    fn skp(&mut self, vx: u8) {
        if self.keys & (0x01 << vx) > 0 {
            self.pc += 2;
        }
    }

    // SKNP VX — EXA1
    fn sknp(&mut self, vx: u8) {
        if self.keys & (0x01 << vx) == 0 {
            self.pc += 2;
        }
    }

    // LD VX, DT — FX07
    fn ld_dt_to_vx(&mut self, vx: u8) {
        self.registers[vx as usize] = self.dt;
    }

    // // LD VX, K — FX0A
    fn ld_k(&mut self, key: u8, vx: u8) {
        self.registers[vx as usize] = key;
    }

    // LD DT, VX — FX15
    fn ld_vx_to_dt(&mut self, vx: u8) {
        self.dt = self.registers[vx as usize];
    }

    // LD ST, VX — FX18
    fn ld_st(&mut self, vx: u8) {
        self.st = self.registers[vx as usize];
    }

    // ADD I, VX — FX1E
    fn add(&mut self, vx: u8) {
        self.ix += self.registers[vx as usize] as u16;
    }

    // LD F, VX — FX29
    fn ld_f(&mut self, vx: u8) {
        self.ix = self.registers[vx as usize] as u16 * 0x05;
    }

    // LD B, VX — FX33
    fn ld_b(&mut self, vx: u8) {
        // get hundreds, tens and ones
        let h = self.registers[vx as usize] as u16 / 100;
        let t = (self.registers[vx as usize] as u16 - h * 100) / 10;
        let o = (self.registers[vx as usize] as u16 - h * 100 - t) * 10;

        self.ram[self.ix as usize] = h as u8;
        self.ram[self.ix as usize + 1] = t as u8;
        self.ram[self.ix as usize + 2] = o as u8;
    }

    // LD [I], VX — FX55
    fn ld_to_i(&mut self, vx: u8) {
        for reg in 0..vx {
            self.ram[self.ix as usize + reg as usize] = self.registers[reg as usize]
        }
    }

    // LD VX, [I] — FX65
    fn ld_from_i(&mut self, vx: u8) {
        for reg in 0..vx {
            self.registers[reg as usize] = self.ram[self.ix as usize + reg as usize]
        }
    }
}

impl ApplicationHandler for Emulator {
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        match self.key_input {
            KeyPressState::Idle => {}
            KeyPressState::Waiting(_) => return,
            KeyPressState::Set(key, reg) => {
                self.core.ld_k(key, reg);
                self.key_input = KeyPressState::Idle
            }
        }
        loop {
            let instr = [
                self.core.ram[self.core.pc as usize + 1],
                self.core.ram[self.core.pc as usize],
            ];
            let next_instruction = u16::from_le_bytes(instr);
            self.core.pc += 2;
            if (next_instruction & 0xF0FF) == 0xF00A {
                self.key_input = KeyPressState::Waiting(reg_x!(next_instruction) as u8);
                return;
            }
            if execute(next_instruction, &mut self.core) {
                self.window.as_ref().unwrap().request_redraw();
                return;
            }
        }
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_inner_size(winit::dpi::LogicalSize::new(
                        self.core.width as f64,  // * 8.0,
                        self.core.height as f64, // * 8.0,
                    ))
                    .with_min_inner_size(winit::dpi::LogicalSize::new(
                        self.core.width as f64 * 8.0,
                        self.core.height as f64 * 8.0,
                    )),
            )
            .unwrap();
        let pixels = {
            let window_size = window.inner_size();
            let surface_texture =
                SurfaceTexture::new(window_size.width, window_size.height, &window);
            Pixels::new(
                self.core.width as u32,
                self.core.height as u32,
                surface_texture,
            )
            .unwrap()
        };
        self.window = Some(window);
        self.pixels = Some(pixels);
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            winit::event::WindowEvent::RedrawRequested => {
                let px = self.pixels.as_mut().unwrap();
                let frame = px.frame_mut();
                draw(&self.core, frame);
                if let Err(err) = px.render() {
                    panic!("pixels.render failed: {}", err);
                }
            }
            winit::event::WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                if let KeyPressState::Waiting(reg) = self.key_input {
                    self.key_input = {
                        if let winit::keyboard::Key::Character(key_char) =
                            event.logical_key.as_ref()
                        {
                            match key_char {
                                "1" => KeyPressState::Set(1, reg),
                                "2" => KeyPressState::Set(2, reg),
                                "3" => KeyPressState::Set(3, reg),
                                "4" => KeyPressState::Set(0xc, reg),
                                "q" => KeyPressState::Set(4, reg),
                                "w" => KeyPressState::Set(5, reg),
                                "e" => KeyPressState::Set(6, reg),
                                "r" => KeyPressState::Set(0xd, reg),
                                "a" => KeyPressState::Set(7, reg),
                                "s" => KeyPressState::Set(8, reg),
                                "d" => KeyPressState::Set(9, reg),
                                "f" => KeyPressState::Set(0xe, reg),
                                "z" => KeyPressState::Set(0xa, reg),
                                "x" => KeyPressState::Set(0, reg),
                                "c" => KeyPressState::Set(0xb, reg),
                                "v" => KeyPressState::Set(0xf, reg),
                                _ => KeyPressState::Waiting(reg),
                            }
                        } else {
                            KeyPressState::Waiting(reg)
                        }
                    };
                }
            }
            _ => {}
        }
    }
}

fn execute(opcode: u16, machine: &mut Core) -> bool {
    if opcode == 0x00E0 {
        machine.cls();
        return true;
    }
    if (opcode & 0xF000) == 0xD000 {
        machine.drw(reg_x!(opcode), reg_y!(opcode), (opcode & BITMASK_N) as u8);
        return true;
    }
    if opcode == 0x00EE {
        machine.ret();
    } else if (opcode & 0xF000) == 0x1000 {
        machine.jmp(opcode & BITMASK_NNN);
    } else if (opcode & 0xF000) == 0x2000 {
        machine.call(opcode & BITMASK_NNN);
    } else if (opcode & 0xF000) == 0x3000 {
        machine.se_n(reg_x!(opcode), (opcode & BITMASK_NN) as u8);
    } else if (opcode & 0xF000) == 0x4000 {
        machine.sne(reg_x!(opcode), (opcode & BITMASK_NN) as u8);
    } else if (opcode & 0xF00F) == 0x5000 {
        machine.se((reg_x!(opcode)) as u8, reg_y!(opcode))
    } else if (opcode & 0xF000) == 0x6000 {
        machine.ld(reg_x!(opcode), (opcode & BITMASK_NN) as u8);
    } else if (opcode & 0xF000) == 0x7000 {
        machine.add_n(reg_x!(opcode), (opcode & BITMASK_NN) as u8);
    } else if (opcode & 0xF00F) == 0x8000 {
        machine.ld_vx_vy(reg_x!(opcode), reg_y!(opcode));
    } else if (opcode & 0xF00F) == 0x8001 {
        machine.or(reg_x!(opcode), reg_y!(opcode));
    } else if (opcode & 0xF00F) == 0x8002 {
        machine.and(reg_x!(opcode), reg_y!(opcode));
    } else if (opcode & 0xF00F) == 0x8003 {
        machine.xor(reg_x!(opcode), reg_y!(opcode));
    } else if (opcode & 0xF00F) == 0x8004 {
        machine.add_vx_vy(reg_x!(opcode), reg_y!(opcode));
    } else if (opcode & 0xF00F) == 0x8005 {
        machine.sub(reg_x!(opcode), reg_y!(opcode));
    } else if (opcode & 0xF00F) == 0x8006 {
        machine.shr(reg_x!(opcode));
    } else if (opcode & 0xF00F) == 0x8007 {
        machine.subn(reg_x!(opcode), reg_y!(opcode));
    } else if (opcode & 0xF00F) == 0x800E {
        machine.shl(reg_x!(opcode));
    } else if (opcode & 0xF00F) == 0x9000 {
        machine.sne_vx_vy(reg_x!(opcode), reg_y!(opcode));
    } else if (opcode & 0xF000) == 0xA000 {
        machine.ld_i((opcode & BITMASK_NNN) as u16);
    } else if (opcode & 0xF000) == 0xB000 {
        machine.jmp_v0((opcode & BITMASK_NNN) as u16);
    } else if (opcode & 0xF000) == 0xC000 {
        machine.rnd(reg_x!(opcode), (opcode & BITMASK_NN) as u8);
    } else if (opcode & 0xF0FF) == 0xE09E {
        machine.skp(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xE0A1 {
        machine.sknp(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xF007 {
        machine.ld_dt_to_vx(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xF015 {
        machine.ld_vx_to_dt(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xF018 {
        machine.ld_st(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xF01E {
        machine.add(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xF029 {
        machine.ld_f(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xF033 {
        machine.ld_b(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xF055 {
        machine.ld_to_i(reg_x!(opcode));
    } else if (opcode & 0xF0FF) == 0xF065 {
        machine.ld_from_i(reg_x!(opcode));
    } else {
        panic!("idk opcode {:#06}", opcode);
    }
    false
}

fn load_program(file_path: &str) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn main() {
    let program = load_program("/Users/mikeo/Downloads/Chrome Downloads/1-chip8-logo.ch8")
        .expect("failed to load program");
    // let program = load_program("/Users/mikeo/Downloads/Chrome Downloads/IBM Logo.ch8").expect("failed to load program");
    // let program = load_program("/Users/mikeo/Downloads/Chrome Downloads/test_opcode.ch8").expect("failed to load program");
    // let program: Vec<u8> = vec![
    //     0xa0, 0x2d,
    //     0x62, 0x10,
    //     0x63, 0x05,
    //     0xd2, 0x35,
    //     0x12, 0x08,
    // ];

    let event_loop = EventLoop::new().unwrap();
    let core = Core::new(64, 32, &program);

    let mut emu = Emulator {
        key_input: KeyPressState::Idle,
        pixels: None,
        core,
        window: None,
    };
    event_loop.run_app(&mut emu).expect("failed to run app");
}
